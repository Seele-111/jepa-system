#!/usr/bin/env python3
"""Explicitly acquire pinned upstream sources and rebuild baseline checkpoints.

Nothing downloads at demo startup. --download-checkpoints is an explicit large
transfer; --sources-only avoids it. Third-party resources remain outside Git and
under their own licenses. Existing files are verified, never overwritten.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from urllib.request import urlopen
import zipfile

from jepa_runtime import ROOT
from published_models import digest_file, registry, source_tree_digest


def download(url, target, expected_sha=None):
    target = Path(target)
    if target.exists():
        if expected_sha and target.is_file() and digest_file(target) == expected_sha:
            return target
        raise ValueError("existing download target is not the expected resource; it was not overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading " + target.name, flush=True)
    with urlopen(url, timeout=120) as response, target.open("xb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if expected_sha and digest_file(target) != expected_sha:
        raise ValueError("download SHA-256 mismatch; keep the rejected file for diagnosis, do not change the registry")
    return target


def prepare_source(kind, directory, claims):
    name = "vjepa2" if kind == "vjepa" else "ijepa"
    target = directory / name
    entry = claims[kind + "_source"]
    pending = target / ".jepa-preparation-incomplete"
    if target.exists():
        if pending.exists() or pending.is_symlink() or not (target / "LICENSE").is_file() or source_tree_digest(target) != entry["source_tree_sha256"]:
            raise ValueError("existing source directory differs from the pinned version; it was not overwritten")
        return target
    archive = directory / "downloads" / (name + "-" + entry["revision"] + ".zip")
    # A source archive can be reused only if its extracted source fingerprint
    # passes; no archive is trusted merely because its filename exists.
    if not archive.exists():
        download(entry["repository"] + "/archive/" + entry["revision"] + ".zip", archive)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        roots = {PurePosixPath(item.filename).parts[0] for item in members if item.filename}
        if len(roots) != 1:
            raise ValueError("unexpected upstream archive layout")
        target.mkdir(parents=True, exist_ok=False)
        # A rejected or interrupted extraction must not become reusable just
        # because its Python subset already happens to match the source digest.
        with pending.open("x", encoding="utf-8") as handle:
            handle.write("Source preparation has not completed validation.\n")
        for item in members:
            parts = PurePosixPath(item.filename).parts
            if PurePosixPath(item.filename).is_absolute() or ".." in parts or "\\" in item.filename:
                raise ValueError("unsafe path in upstream archive")
            if len(parts) <= 1:
                continue
            # Only inference source trees and upstream notices are needed.
            # Omit training configs, which contain case-colliding filenames on
            # Windows; do not silently overwrite one with another.
            relative = parts[1:]
            root_notice = len(relative) == 1 and (relative[0].startswith(("LICENSE", "NOTICE", "COPYING", "README", "requirements")) or relative[0] in {"pyproject.toml", "setup.py"})
            if relative[0] not in {"src", "app"} and not root_notice:
                continue
            destination = (target.joinpath(*parts[1:])).resolve()
            if not destination.is_relative_to(target.resolve()) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("unsafe link in upstream archive")
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, destination.open("xb") as handle:
                    shutil.copyfileobj(source, handle)
    if source_tree_digest(target) != entry["source_tree_sha256"] or not (target / "LICENSE").is_file():
        raise ValueError("downloaded upstream source/license verification failed")
    pending.unlink()
    print(kind + " source verified", flush=True)
    return target


def checked_file(path, claim):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or digest_file(path) != claim["sha256"]:
        raise ValueError("input checkpoint is missing or its SHA-256 does not match the pinned baseline")
    return path


def convert_checkpoint(full, target, kind, claims, conversion):
    if target.exists():
        return checked_file(target, claims["vjepa_encoder" if kind == "vjepa" else "ijepa_checkpoint"])
    import torch
    if torch.__version__ != conversion["torch_version"]:
        raise ValueError("checkpoint conversion requires the pinned Torch version for file-level identity")
    target.parent.mkdir(parents=True, exist_ok=True)
    source = torch.load(full, weights_only=True, mmap=True, map_location="cpu")
    if kind == "vjepa":
        state = {"encoder": {key.replace("module.", "").replace("backbone.", ""): value
                             for key, value in source[conversion["vjepa_encoder_branch"]].items()}}
    else:
        state = {branch: {key.replace("module.", ""): value.to(torch.bfloat16)
                          for key, value in source[branch].items()}
                 for branch in conversion["ijepa_branches"]}
    # Filename and Torch version are pinned: changing zip serialization must not
    # be "fixed" by editing the model/resource SHA claims.
    claim = claims["vjepa_encoder" if kind == "vjepa" else "ijepa_checkpoint"]
    # Preserve the basename required by Torch's zip serialization, while never
    # truncating a target (including a late-created hardlink) after verification.
    with tempfile.TemporaryDirectory(prefix=".jepa-conversion-", dir=target.parent) as temporary:
        stage = Path(temporary).resolve()
        if not stage.is_relative_to(target.parent.resolve()):
            raise ValueError("conversion staging escaped the intended resource directory")
        staged = stage / target.name
        torch.save(state, staged)
        checked_file(staged, claim)
        with staged.open("rb") as source_file, target.open("xb") as output_file:
            shutil.copyfileobj(source_file, output_file, length=1024 * 1024)
    del source, state
    gc.collect()
    return checked_file(target, claim)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "resources")
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--download-checkpoints", action="store_true")
    parser.add_argument("--accept-noncommercial", action="store_true",
                        help="acknowledge I-JEPA source's CC BY-NC license; review checkpoint terms yourself")
    parser.add_argument("--vjepa-full", type=Path)
    parser.add_argument("--ijepa-full", type=Path)
    parser.add_argument("--r3d", type=Path)
    parser.add_argument("--write-config", type=Path, help="write a NEW ignored runtime TOML, never overwrite one")
    parser.add_argument("--transport", choices=("native", "wsl"), default="native")
    args = parser.parse_args(argv)
    if not args.accept_noncommercial:
        parser.error("read upstream licenses, then explicitly use --accept-noncommercial; this does not grant commercial rights")
    if args.sources_only and (args.download_checkpoints or args.write_config):
        parser.error("--sources-only does not download checkpoints or write a full-mode configuration")
    if not args.sources_only and not args.download_checkpoints and not all((args.vjepa_full, args.ijepa_full, args.r3d)):
        parser.error("provide all three existing official checkpoints, or explicitly use --download-checkpoints")
    if args.write_config and args.write_config.exists():
        parser.error("configuration already exists; it was not overwritten")
    directory = args.directory.expanduser().resolve()
    release = registry()
    claims = release["resources"]
    sources = {kind: prepare_source(kind, directory, claims) for kind in ("vjepa", "ijepa")}
    if args.sources_only:
        print(json.dumps({"status": "sources_verified", "source_paths": {key: str(value) for key, value in sources.items()}}))
        return 0
    def checkpoint(provided, name, filename):
        if provided:
            return checked_file(provided, claims[name])
        target = directory / "checkpoints" / filename
        return download(claims[name]["source_url"], target, claims[name]["sha256"])
    vfull = checkpoint(args.vjepa_full, "vjepa_predictor", "vjepa2_1_vitg_384.pt")
    ifull = checkpoint(args.ijepa_full, "ijepa_full", "IN22K-vit.g.16-600e.pth.tar")
    r3d = checkpoint(args.r3d, "r3d_checkpoint", "r3d_18-b3b3357e.pth")
    encoder = convert_checkpoint(vfull, directory / "checkpoints" / "encoder_only.pt", "vjepa", claims, release["conversion"])
    ijepa = convert_checkpoint(ifull, directory / "checkpoints" / "ijepa_true_slim_bf16.pt", "ijepa", claims, release["conversion"])
    resources = {"vjepa_source": sources["vjepa"], "ijepa_source": sources["ijepa"], "vjepa_encoder": encoder,
                 "vjepa_predictor": vfull, "ijepa_checkpoint": ijepa, "r3d_checkpoint": r3d}
    if args.write_config:
        # JSON basic strings are a valid subset for these TOML path values.
        text = ('[runtime]\ntransport = ' + json.dumps(args.transport) + '\nworker_port = 5004\nretain_models = true\n')
        interpreter_key = "wsl_python" if args.transport == "wsl" else "python"
        interpreter = "python3" if args.transport == "wsl" and os.name == "nt" else sys.executable
        text += interpreter_key + " = " + json.dumps(interpreter) + "\n\n[resources]\n"
        text += "\n".join(key + " = " + json.dumps(str(value)) for key, value in resources.items())
        text += '\n\n[demo]\ndefault_algorithm = "optimized"\n'
        args.write_config.parent.mkdir(parents=True, exist_ok=True)
        with args.write_config.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(json.dumps({"status": "resources_verified", "resource_paths": {key: str(value) for key, value in resources.items()},
                      "third_party_weights_distributed": False}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print("Resource preparation failed: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
