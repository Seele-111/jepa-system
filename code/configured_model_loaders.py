"""Configured, source-verified model isolation without changing core extractors."""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import importlib.util
from pathlib import Path
import sys
import types
from importlib.machinery import ModuleSpec

from jepa_runtime import ROOT, settings
from published_models import digest_file, registry, validate_source


def _exec_module(alias, path):
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError("model adapter cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module



def _memory_efficient_vjepa_class(loader):
    """Change checkpoint staging, not model math, weights or preprocessing.

    The historical loader temporarily copies the FP32 state to CUDA beside the
    BF16 encoder. mmap on CPU avoids that redundant GPU allocation; strict copy
    into the same BF16 module yields the same inference tensors.
    """
    class ConfiguredVJEPASurprise(loader.VJEPASurprise):
        def _load_encoder(self):
            if self.encoder is not None:
                return
            from src.hub.backbones import vjepa2_1_vit_giant_384
            torch = loader.torch
            encoder, unused_predictor = vjepa2_1_vit_giant_384(pretrained=False)
            del unused_predictor
            encoder = encoder.to(loader.DEVICE, dtype=torch.bfloat16).eval()
            state = torch.load(loader.ENCODER_CKPT, map_location="cpu", weights_only=True, mmap=True)
            encoder.load_state_dict(state["encoder"], strict=True)
            del state
            self.encoder = encoder
            torch.cuda.empty_cache()
    return ConfiguredVJEPASurprise


@contextmanager
def isolated_legacy_module(kind):
    if kind not in ("vjepa", "ijepa"):
        raise ValueError("unknown JEPA model kind")
    cfg = settings()
    source = cfg.resources[kind + "_source"]
    validate_source(kind, source)
    code = ROOT / "code"
    aliases = {"_jepa_configured_" + kind, "ijepa_predict", "_jepa_components"}
    def scoped(name):
        return name in aliases or name in {"src", "app"} or name.startswith(("src.", "app."))
    saved = {name: module for name, module in list(sys.modules.items()) if scoped(name)}
    paths, bytecode = sys.path[:], sys.dont_write_bytecode
    try:
        for name in saved:
            del sys.modules[name]
        sys.path[:0] = [str(source), str(code)]
        sys.dont_write_bytecode = True
        # Upstream src/app may be PEP 420 namespace packages. Build an explicit
        # package path so an editable install of another upstream cannot merge
        # its private src tree into this verified resource scope.
        for name in ("src", "app"):
            package_root = source / name
            if not package_root.is_dir():
                continue
            initializer = package_root / "__init__.py"
            if initializer.is_file():
                package = _exec_module(name, initializer)
                package.__path__ = [str(package_root)]
            else:
                package = types.ModuleType(name)
                package.__package__ = name
                package.__path__ = [str(package_root)]
                package.__spec__ = ModuleSpec(name, loader=None, is_package=True)
                package.__spec__.submodule_search_locations = package.__path__
                sys.modules[name] = package
        if kind == "ijepa":
            path = code / "jepa_model_components.py"
            if digest_file(path) != registry()["implementation"][path.name]["sha256"]:
                raise ValueError("I-JEPA component adapter differs from the reviewed release")
            components = _exec_module("_jepa_components", path)
            sys.modules["ijepa_predict"] = components
        path = code / (kind + "_predictor.py")
        if digest_file(path) != registry()["implementation"][path.name]["sha256"]:
            raise ValueError("legacy model loader differs from the reviewed release")
        loader = _exec_module("_jepa_configured_" + kind, path)
        # Checkpoint content identity is still checked by the worker before load.
        if kind == "vjepa":
            loader.ENCODER_CKPT = str(cfg.resources["vjepa_encoder"])
            loader.FULL_CKPT = str(cfg.resources["vjepa_predictor"])
            loader.VJEPASurprise = _memory_efficient_vjepa_class(loader)
        else:
            loader.IJEPA_ROOT = source
            loader.CKPT_FULL = loader.CKPT_SLIM_BF16 = cfg.resources["ijepa_checkpoint"]
        sys.path[:] = [str(source), str(code), *paths]
        yield loader
    finally:
        for name in list(sys.modules):
            if scoped(name):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = paths
        sys.dont_write_bytecode = bytecode


def verify_checkpoint_resources(names=("vjepa_encoder", "vjepa_predictor", "ijepa_checkpoint")):
    cfg, claims = settings(), registry()["resources"]
    for name in names:
        path = cfg.resources[name]
        if not path.is_file() or digest_file(path) != claims[name]["sha256"]:
            raise ValueError(name + " is missing or its SHA-256 differs from the reviewed resource")


class LegacyImageScorer:
    """Use genuine historical I-JEPA compute() inside its own upstream scope."""
    def __init__(self, mask_seed=0):
        self.seed = mask_seed

    def compute(self, frames, batch_size=2):
        with isolated_legacy_module("ijepa") as loader:
            scorer = loader.IJEPASurprise(mask_seed=self.seed)
            try:
                return scorer.compute(frames, batch_size=batch_size)
            finally:
                scorer._free_models()
