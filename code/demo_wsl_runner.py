#!/usr/bin/env python3
"""Seeded legacy launcher; optional configured, source-verified resource scopes.

The original pipeline/seed positional CLI remains accepted. The product launcher
adds --configured-resources to use this checkout rather than a private WSL tree.
"""
import argparse
import importlib.util
import os
import random
import runpy
import sys
from pathlib import Path
import types

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--configured-resources", action="store_true")
    parser.add_argument("pipeline")
    parser.add_argument("seed", type=int)
    args, forwarded = parser.parse_known_args()
    if args.config:
        os.environ["JEPA_CONFIG"] = str(Path(args.config).resolve())
    pipeline, seed = str(Path(args.pipeline).resolve()), args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    sys.path.insert(0, str(Path(pipeline).parent))
    sys.argv = [pipeline, *forwarded]
    print(f"[DemoRunner] Fixed random seed: {seed}", flush=True)
    if not args.configured_resources:
        runpy.run_path(pipeline, run_name="__main__")
        return
    from configured_model_loaders import isolated_legacy_module, LegacyImageScorer, verify_checkpoint_resources
    from jepa_runtime import ROOT, settings
    from published_models import digest_file, registry
    cfg = settings()
    expected = registry()["implementation"]["detect_and_report_v4.py"]["sha256"]
    if digest_file(pipeline) != expected:
        raise ValueError("configured legacy pipeline is not the reviewed implementation")
    verify_checkpoint_resources()
    spec = importlib.util.spec_from_file_location("_jepa_configured_pipeline", pipeline)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_index = forwarded.index("--output") + 1
    output = Path(forwarded[output_index]).resolve()
    # The genuine modes use per-video normalization; no private normal stats or
    # dataset cache is copied. Any legacy scratch artifact stays with this job.
    module.DATA_ROOT = str(output / "legacy-state")
    module.CACHE_DIR = str(output / "legacy-feature-cache")
    module.VJEPA_CKPT = str(cfg.resources["vjepa_encoder"])
    module.GLOBAL_STATS_PATH = str(output / "unprovided-global-stats.pt")
    module.GLOBAL_STATS_PATH_V2 = str(output / "unprovided-global-stats-v2.pt")
    saved = {key: sys.modules.get(key) for key in ("vjepa_predictor", "ijepa_predictor")}
    try:
        with isolated_legacy_module("vjepa") as loader:
            sys.modules["vjepa_predictor"] = loader
            image = types.ModuleType("ijepa_predictor")
            image.IJEPASurprise = LegacyImageScorer
            sys.modules["ijepa_predictor"] = image
            module.main()
    finally:
        for key, original in saved.items():
            if original is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = original


if __name__ == "__main__":
    main()
