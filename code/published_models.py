"""Validation and address-only relocation of explicitly reviewed public readouts.

A public profile is pinned in the release registry before any address is resolved.
Only resource addresses may move. Checkpoint hashes, preprocessing, runtime
versions, feature order and core extractor hashes are never relaxed. Original
research bundles still use their original validation path unchanged.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SCHEMA = "jepa-public-readout-v1"
REGISTRY = ROOT / "models" / "public" / "registry.json"


def digest_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest_json(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def registry():
    result = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if result.get("schema_version") != PUBLIC_SCHEMA:
        raise ValueError("unknown public resource registry")
    return result


def validate_bundle(bundle, path):
    claim = bundle.get("publication")
    if not isinstance(claim, dict) or claim.get("schema_version") != PUBLIC_SCHEMA:
        raise ValueError("unknown public readout schema")
    entry = registry()["models"].get(claim.get("model_id"))
    if entry is None or digest_file(path) != entry["file_sha256"]:
        raise ValueError("public model file differs from the reviewed release")
    payload = {key: value for key, value in bundle.items() if key != "publication"}
    if digest_json(payload) != entry["inference_payload_sha256"] or claim["inference_payload_sha256"] != entry["inference_payload_sha256"]:
        raise ValueError("public inference payload identity mismatch")
    if claim.get("source_bundle_sha256") != entry["source_bundle_sha256"]:
        raise ValueError("public source model provenance mismatch")
    validate_profiles(bundle["feature_profiles"])


def validate_profiles(profiles):
    identity = digest_json(profiles)
    if not any(identity == entry["feature_profiles_sha256"] for entry in registry()["models"].values()):
        raise ValueError("public feature profile differs from the reviewed release")
    # Pin implementation bytes as well as model-level feature/profile identities.
    motion = profiles.get("motion")
    if motion and digest_file(ROOT / "code" / "optimized_motion_features.py") != motion["code_sha256"]:
        raise ValueError("motion extractor differs from the released model")
    corrected = profiles.get("corrected")
    if corrected:
        for key, name in (("extractor_sha256", "optimized_jepa_extractor.py"),):
            if digest_file(ROOT / "code" / name) != corrected[key]:
                raise ValueError("corrected extractor/descriptor provenance mismatch")
    if corrected:
        # adapter_sha256 records the historical collection script, not a new
        # claim about its full current bytes. Descriptor parity was reviewed
        # separately, and the current inference implementation is pinned.
        current = registry()["implementation"]["build_optimization_jepa.py"]
        if digest_file(ROOT / "code" / "build_optimization_jepa.py") != current["sha256"]:
            raise ValueError("released descriptor implementation changed")
    rgb = profiles.get("rgb")
    if rgb:
        unsigned = {key: value for key, value in rgb.items() if key != "profile_id"}
        if digest_json(unsigned) != rgb["profile_id"]:
            raise ValueError("public RGB profile_id mismatch")
        if digest_file(ROOT / "code" / "extract_rgb_r3d_features.py") != rgb["preprocess"]["baseline_sha256"]:
            raise ValueError("RGB spatial baseline provenance mismatch")


def resolve_profiles(profiles):
    """Return a new exact profile with only reviewed resource addresses relocated."""
    if not any(str(profile.get("checkpoint_path", "")).startswith("resource://") or
               any(str(item.get("path", "")).startswith("resource://") for item in profile.get("checkpoints", {}).values())
               for profile in profiles.values() if isinstance(profile, dict)):
        return profiles
    validate_profiles(profiles)
    from jepa_runtime import settings
    cfg = settings()
    resolved = copy.deepcopy(profiles)
    rgb = resolved.get("rgb")
    if rgb:
        if rgb["checkpoint_path"] != "resource://r3d_checkpoint" or rgb["preprocess"]["baseline_path"] != "resource://rgb_baseline":
            raise ValueError("unreviewed RGB relocation address")
        rgb["checkpoint_path"] = str(cfg.resources["r3d_checkpoint"])
        rgb["preprocess"]["baseline_path"] = str((ROOT / "code" / "extract_rgb_r3d_features.py").resolve())
        rgb["profile_id"] = digest_json({key: value for key, value in rgb.items() if key != "profile_id"})
    corrected = resolved.get("corrected")
    if corrected:
        for claim, resource in (("vjepa_encoder", "vjepa_encoder"), ("vjepa_predictor", "vjepa_predictor"), ("ijepa", "ijepa_checkpoint")):
            item = corrected["checkpoints"][claim]
            if item["path"] != "resource://" + resource or item["sha256"] != registry()["resources"][resource]["sha256"]:
                raise ValueError("unreviewed JEPA checkpoint relocation")
            item["path"] = str(cfg.resources[resource])
    return resolved


def source_tree_digest(root):
    """Fingerprint upstream inference Python sources, independent of directory."""
    root = Path(root).resolve()
    names = sorted(path.relative_to(root).as_posix() for folder in ("src", "app")
                   for path in (root / folder).rglob("*.py") if path.is_file())
    if not names:
        raise ValueError("configured upstream is missing Python inference sources")
    result = hashlib.sha256()
    for name in names:
        result.update((name + "\0" + digest_file(root / name) + "\n").encode())
    return result.hexdigest()


def validate_source(kind, root):
    root = Path(root).resolve()
    pending = root / ".jepa-preparation-incomplete"
    if pending.exists() or pending.is_symlink() or not (root / "LICENSE").is_file():
        raise ValueError("configured upstream is incomplete or missing its LICENSE; prepare a verified resource directory")
    entry = registry()["resources"][kind + "_source"]
    if source_tree_digest(root) != entry["source_tree_sha256"]:
        raise ValueError(kind + " upstream inference sources differ from the pinned release; see resource setup guide")
    return entry
