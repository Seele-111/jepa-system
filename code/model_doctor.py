#!/usr/bin/env python3
"""Check package readiness or explicitly verify the configured model resources.

Default checks stay CPU/lightweight. --model-runtime executes the resource check
using the configured native/WSL interpreter; it does not load model weights,
read user videos, download files or launch a persistent worker.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from jepa_runtime import ROOT, model_command, settings
from optimized_locator import load_bundle
from published_models import registry, validate_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--check-resources", action="store_true")
    parser.add_argument("--model-runtime", action="store_true")
    args = parser.parse_args(argv)
    if args.config:
        os.environ["JEPA_CONFIG"] = str(Path(args.config).resolve())
    if args.model_runtime:
        proc = subprocess.run(model_command(Path(__file__), ["--check-resources"]),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=600)
        from demo_detector import _decode_process_bytes
        print(_decode_process_bytes(proc.stdout), end="")
        if proc.returncode:
            print(_decode_process_bytes(proc.stderr), file=sys.stderr, end="")
        return proc.returncode
    cfg = settings()
    result = {"status": "ok", "package_models": {}, "resource_readiness_checked": args.check_resources,
              "transport": cfg.transport, "errors": []}
    for name, path in (("optimized", cfg.full_bundle), ("optimized_fast", cfg.motion_bundle)):
        try:
            bundle = load_bundle(path)
            reviewed = bool(bundle.get("publication"))
            result["package_models"][name] = {"readable": True, "verified": reviewed, "publication": reviewed,
                                              "validation_level": "reviewed_public_release" if reviewed else "legacy_schema_only"}
        except (OSError, ValueError, KeyError) as exc:
            result["package_models"][name] = {"verified": False}
            result["errors"].append({"component": name, "code": "bundle_invalid", "message": str(exc)})
    if args.check_resources:
        from configured_model_loaders import verify_checkpoint_resources
        try:
            for kind in ("vjepa", "ijepa"):
                validate_source(kind, cfg.resources[kind + "_source"])
            verify_checkpoint_resources(("vjepa_encoder", "vjepa_predictor", "ijepa_checkpoint", "r3d_checkpoint"))
            import torch
            import torchvision
            import cv2
            import numpy as np
            actual = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                      "numpy": np.__version__, "opencv": cv2.__version__, "python": sys.version.split()[0]}
            expected = load_bundle(cfg.full_bundle)["feature_profiles"]["rgb"]["runtime_versions"]
            result["runtime_versions"] = actual
            if actual != expected:
                raise ValueError("runtime version profile differs from the baseline; prepare the pinned environment, do not edit the model profile")
            result["cuda_available"] = torch.cuda.is_available()
            if not result["cuda_available"]:
                raise ValueError("genuine JEPA requires CUDA; CPU learned motion is a separate mode")
            result["resources_verified"] = True
        except (OSError, ValueError, ImportError, KeyError) as exc:
            result["resources_verified"] = False
            result["errors"].append({"component": "gpu_resources", "code": "resources_not_ready", "message": str(exc)})
    if result["errors"]:
        result["status"] = "error"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
