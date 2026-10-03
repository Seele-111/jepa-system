"""Local, explicit runtime/resource configuration; no private machine defaults.

A copied ``jepa-runtime.example.toml`` can be selected with JEPA_CONFIG.
Resource paths are interpreted by the model runtime (Linux paths for WSL).
Imports are lightweight and never download resources or load Torch.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import sys
import tomllib
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
MODEL_ENV = {"JEPA_TRANSPORT", "JEPA_WORKER_PORT", "JEPA_WSL_DISTRO", "JEPA_WSL_PYTHON",
             "JEPA_DEMO_DATA", "JEPA_FULL_BUNDLE", "JEPA_MOTION_BUNDLE"}
MODEL_ENV.update("JEPA_" + name.upper() for name in ("vjepa_source", "ijepa_source", "vjepa_encoder",
                                                    "vjepa_predictor", "ijepa_checkpoint", "r3d_checkpoint"))


def config_path() -> Path | None:
    explicit = os.environ.get("JEPA_CONFIG")
    path = Path(explicit).expanduser() if explicit else ROOT / "jepa-runtime.toml"
    if explicit and not path.is_file():
        raise ValueError("JEPA_CONFIG must point to an existing local TOML file")
    return path.resolve() if path.is_file() else None


def read_config() -> dict:
    path = config_path()
    if path is None:
        return {}
    value = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    if any(key not in {"runtime", "resources", "demo"} for key in value):
        raise ValueError("unknown runtime configuration section")
    allowed = {
        "runtime": {"transport", "python", "wsl_distro", "wsl_python", "worker_port", "retain_models"},
        "resources": {"vjepa_source", "ijepa_source", "vjepa_encoder", "vjepa_predictor", "ijepa_checkpoint", "r3d_checkpoint"},
        "demo": {"data_root", "full_bundle", "motion_bundle", "default_algorithm"},
    }
    for section, fields in value.items():
        if not isinstance(fields, dict) or set(fields) - allowed[section]:
            raise ValueError("unknown field in runtime configuration: " + section)
    return value


def _local_path(value: str, root: Path = ROOT) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("resource paths must be nonempty local filesystem paths")
    if os.name != "nt":
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
        if match:
            value = "/mnt/" + match[1].lower() + "/" + match[2].replace("\\", "/")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


@dataclass(frozen=True)
class RuntimeConfig:
    transport: str
    python: str
    wsl_distro: str
    wsl_python: str
    worker_port: int
    retain_models: bool
    resources: dict[str, Path]
    data_root: Path
    full_bundle: Path
    motion_bundle: Path
    default_algorithm: str
    config_id: str

    @property
    def worker_url(self) -> str:
        return "http://127.0.0.1:" + str(self.worker_port)


def settings() -> RuntimeConfig:
    value = read_config()
    runtime, resources, demo = (value.get(key, {}) for key in ("runtime", "resources", "demo"))
    transport = os.environ.get("JEPA_TRANSPORT", runtime.get("transport", "native"))
    if transport not in {"native", "wsl"}:
        raise ValueError("runtime.transport must be native or wsl")
    port = int(os.environ.get("JEPA_WORKER_PORT", runtime.get("worker_port", 5004)))
    if not 1024 <= port <= 65535:
        raise ValueError("worker_port must be between 1024 and 65535")
    if type(runtime.get("retain_models", True)) is not bool:
        raise ValueError("retain_models must be a boolean")
    # These addresses are only defaults for resources a user explicitly prepares.
    resource_dir = ROOT / "resources"
    defaults = {
        "vjepa_source": resource_dir / "vjepa2",
        "ijepa_source": resource_dir / "ijepa",
        "vjepa_encoder": resource_dir / "checkpoints" / "encoder_only.pt",
        "vjepa_predictor": resource_dir / "checkpoints" / "vjepa2_1_vitg_384.pt",
        "ijepa_checkpoint": resource_dir / "checkpoints" / "ijepa_true_slim_bf16.pt",
        "r3d_checkpoint": resource_dir / "checkpoints" / "r3d_18-b3b3357e.pth",
    }
    resolved = {key: _local_path(os.environ.get("JEPA_" + key.upper(), resources.get(key, str(path))))
                for key, path in defaults.items()}
    def bundle(key, original, published):
        supplied = os.environ.get("JEPA_" + key.upper(), demo.get(key))
        return _local_path(supplied) if supplied else (ROOT / "models" / original if (ROOT / "models" / original).is_file()
                                                     else ROOT / "models" / "public" / published)
    path = config_path()
    config_id = hashlib.sha256(path.read_bytes()).hexdigest() if path else "defaults"
    # Env-based configuration is supported, but separate workers must use exactly
    # the same resource/transport overrides; the requester still verifies root/code.
    overrides = {key: val for key, val in os.environ.items()
                 if key in MODEL_ENV}
    if overrides:
        import json
        config_id = hashlib.sha256((config_id + json.dumps(overrides, sort_keys=True)).encode()).hexdigest()
    algorithm = demo.get("default_algorithm", "optimized_fast")
    if algorithm not in {"optimized", "optimized_fast", "legacy"}:
        raise ValueError("unsupported default_algorithm")
    return RuntimeConfig(transport, runtime.get("python", sys.executable),
                         os.environ.get("JEPA_WSL_DISTRO", runtime.get("wsl_distro", "Ubuntu-24.04")),
                         os.environ.get("JEPA_WSL_PYTHON", runtime.get("wsl_python", "python3")), port, runtime.get("retain_models", True), resolved,
                         _local_path(os.environ.get("JEPA_DEMO_DATA", demo.get("data_root", str(ROOT / "output" / "public-samples")))),
                         bundle("full_bundle", "optimized_locator_v1.json", "locator.json"),
                         bundle("motion_bundle", "optimized_motion_locator_v1.json", "motion.json"), algorithm, config_id)


def runtime_path(path: str | Path, cfg: RuntimeConfig | None = None) -> str:
    cfg = cfg or settings()
    raw = str(Path(path).resolve())
    if cfg.transport == "wsl":
        match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
        if match:
            return "/mnt/" + match[1].lower() + "/" + match[2].replace("\\", "/")
    return raw


def host_path(path: str | Path, cfg: RuntimeConfig | None = None) -> Path:
    cfg = cfg or settings()
    if os.name == "nt" and cfg.transport == "wsl":
        match = re.match(r"^/mnt/([a-z])/(.*)$", str(path))
        if match:
            return Path(match[1].upper() + ":\\" + match[2].replace("/", "\\"))
    return Path(path)


def model_command(script: Path, arguments: list[str]) -> list[str]:
    cfg = settings()
    path = config_path()
    args = list(arguments)
    if path is not None:
        args = ["--config", runtime_path(path, cfg), *args]
    if cfg.transport == "wsl":
        overrides = [key + "=" + os.environ[key] for key in sorted(MODEL_ENV) if key in os.environ]
        prefix = ["env", *overrides] if overrides else []
        return ["wsl.exe", "-d", cfg.wsl_distro, "--", *prefix, cfg.wsl_python, "-u", "-B", runtime_path(script, cfg), *args]
    return [cfg.python, "-u", "-B", str(script.resolve()), *args]
