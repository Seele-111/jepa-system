#!/usr/bin/env python3
"""Export reviewed inference-only readouts; never edits the source model.

This is a maintainer tool, not a license/provenance bypass. Only the two known
baseline recipes are supported. The public registry must be reviewed and its
models tested for prediction parity before publishing. Private dataset records,
video identifiers, experiment reports and path-bound profiles are not exported.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

SCHEMA = "jepa-public-readout-v1"
INFERENCE_KEYS = {"schema_version", "recipe", "members", "transform", "frame_model", "video_model",
                  "boundary_models", "decoder", "raw_feature_names", "feature_profiles"}
LEAF_KEYS = {"weight", "recipe", "transform", "frame_model", "video_model", "boundary_models"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_public(source, model_id, resource_claims):
    if source.get("schema_version") != "optimized-video-locator-v1":
        raise ValueError("only the baseline locator schema can be released")
    if model_id == "locator":
        expected = ["corrected_motion_rf", "corrected_motion_et", "rgb_motion_tcn"]
        if [member.get("recipe") for member in source.get("members", [])] != expected:
            raise ValueError("unreviewed default recipe")
    elif model_id == "motion":
        if (source.get("recipe") != "motion_rf" or len(source.get("members", [])) != 1
                or source["members"][0].get("recipe") != "motion_rf" or source["members"][0].get("weight") != 1.0):
            raise ValueError("unreviewed CPU recipe")
    else:
        raise ValueError("unknown model_id")
    # Never silently drop inference-affecting experimental extensions.
    for key in ("calibration", "video_gate", "proposal_verifier", "diagnostic_only"):
        if source.get(key) is not None:
            raise ValueError("unreviewed inference extension: " + key)
    result = {key: copy.deepcopy(value) for key, value in source.items() if key in INFERENCE_KEYS}
    for member in result.get("members", []):
        if set(member) - LEAF_KEYS:
            raise ValueError("unreviewed model member fields")
    profiles = result["feature_profiles"]
    source_profile = profiles.get("rgb", {}).get("profile_id")
    if model_id == "motion":
        # This branch consumes only motion; unused JEPA/RGB provenance is omitted.
        result["feature_profiles"] = profiles = {"motion": profiles["motion"]}
        result["raw_feature_names"] = {"motion": result["raw_feature_names"]["motion"]}
    else:
        rgb = profiles["rgb"]
        original_id = rgb.pop("profile_id")
        if digest(rgb) != original_id:
            raise ValueError("source RGB profile identity is invalid")
        rgb["checkpoint_path"] = "resource://r3d_checkpoint"
        rgb["preprocess"]["baseline_path"] = "resource://rgb_baseline"
        rgb["profile_id"] = digest(rgb)
        claims = {}
        for claim, resource in (("vjepa_encoder", "vjepa_encoder"), ("vjepa_predictor", "vjepa_predictor"),
                                ("ijepa", "ijepa_checkpoint")):
            sha = resource_claims[resource]["sha256"]
            if len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
                raise ValueError("missing checkpoint SHA claim")
            claims[claim] = {"path": "resource://" + resource, "sha256": sha}
        profiles["corrected"]["checkpoints"] = claims
    result["training_role"] = "development-trained readout; not independently calibrated; dataset not distributed"
    result["publication"] = {"schema_version": SCHEMA, "model_id": model_id,
                             "source_rgb_profile_id": source_profile if model_id == "locator" else None,
                             "training_membership": "not_distributed", "inference_payload_sha256": digest(result)}
    # Verify all learned arrays/trees, transforms and decoder values are untouched.
    for key in ("members", "transform", "frame_model", "video_model", "boundary_models", "decoder", "recipe"):
        if source.get(key) != result.get(key):
            raise ValueError("release changed inference state: " + key)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--resource-claims", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output must be a new directory; do not overwrite a reviewed release")
    claims = json.loads(args.resource_claims.read_text(encoding="utf-8-sig"))
    bundles = {}
    for name, path in (("locator", args.full), ("motion", args.motion)):
        source = json.loads(path.read_text(encoding="utf-8-sig"))
        bundles[name] = make_public(source, name, claims)
        bundles[name]["publication"]["source_bundle_sha256"] = sha_file(path)
    args.output.mkdir(parents=True, exist_ok=False)
    registry = {"schema_version": SCHEMA, "models": {}, "resources": claims}
    for name, bundle in bundles.items():
        target = args.output / (name + ".json")
        target.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
        registry["models"][name] = {"file_sha256": sha_file(target),
                                   "inference_payload_sha256": bundle["publication"]["inference_payload_sha256"],
                                   "feature_profiles_sha256": digest(bundle["feature_profiles"]),
                                   "source_bundle_sha256": bundle["publication"]["source_bundle_sha256"]}
    code = Path(__file__).resolve().parent
    registry["implementation"] = {}
    for name in ("build_optimization_jepa.py", "jepa_model_components.py", "vjepa_predictor.py",
                 "ijepa_predictor.py", "detect_and_report_v4.py"):
        registry["implementation"][name] = {"sha256": sha_file(code / name)}
    registry["implementation"]["build_optimization_jepa.py"].update(
        historical_source_sha256=bundles["locator"]["feature_profiles"]["corrected"]["adapter_sha256"],
        review="A changed historical collection script requires separate descriptor parity evidence; never rehash its training provenance.")
    (args.output / "registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
