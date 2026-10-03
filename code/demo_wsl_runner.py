#!/usr/bin/env python3
"""Seeded, isolated launcher for the unchanged historical WSL pipeline."""
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    pipeline, seed = str(Path(sys.argv[1]).resolve()), int(sys.argv[2])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Match launching the pipeline directly, including its module resolution.
    sys.path.insert(0, str(Path(pipeline).parent))
    sys.argv = [pipeline, *sys.argv[3:]]
    print(f"[DemoRunner] Fixed random seed: {seed}", flush=True)
    runpy.run_path(pipeline, run_name="__main__")


if __name__ == "__main__":
    main()