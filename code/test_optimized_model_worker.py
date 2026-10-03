"""CPU/NumPy-only worker tests: injected fake weights/backends, no GPU/downloads.

Run with CUDA_VISIBLE_DEVICES=-1 and python -B. Small temporary fixtures are
removed automatically; HTTP tests use a temporary loopback port in this process.
No production service, background process, dataset scan, or real model is used.
"""
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
import hashlib
import io
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from email.message import Message
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import optimized_model_worker as worker
import optimized_jepa_extractor as jepa


def corrected_profile():
    return {"name": "corrected-true-jepa-absolute-patch-v2",
            "config": asdict(jepa.Config(seed=0, max_frames=32, max_keyframes=8, bidirectional=True, mask_passes=1)),
            "extractor_sha256": worker.sha256_file(Path(jepa.__file__)),
            "adapter_sha256": worker.sha256_file(Path(worker.__file__).with_name("build_optimization_jepa.py")),
            "feature_names": worker.CORRECTED_NAMES.copy(), "legacy_cache_compatible": False,
            "i_mask_reuse_policy": "explicit_batch_seed_reset_and_restore"}


def rgb_profile():
    import optimized_rgb_features as rgb
    profile = {"schema_version": rgb.SCHEMA, "encoder": "torchvision.r3d_18", "weights": "KINETICS400_V1",
               "feature_dim": 512, "clip_length": 16, "window_seconds": 1.0, "anchor_interval_seconds": 0.25,
               "frozen": True, "eval_mode": True, "inference_mode": True,
               "fps_source": "cv2.CAP_PROP_FPS from the source video; invalid FPS is an error",
               "checkpoint_path": str(worker.R3D_CHECKPOINT.resolve()), "checkpoint_sha256": "a" * 64,
               "preprocess": {"color": "OpenCV BGR -> RGB", "input": "uint8 TCHW in [0,255]",
                              "resize_hw": [128, 171], "crop_hw": [112, 112], "crop": "center",
                              "resize_interpolation": "bilinear", "resize_antialias": False,
                              "convert_dtype": "float32 / 255", "mean": [0.43216, 0.394666, 0.37645],
                              "std": [0.22803, 0.22145, 0.216989], "baseline_path": str(rgb.BASELINE.resolve()),
                              "baseline_sha256": worker.sha256_file(rgb.BASELINE)}}
    profile["profile_id"] = hashlib.sha256(worker.canonical_json(profile).encode()).hexdigest()
    return profile


class FakeVBackend:
    grid = 2
    def __init__(self):
        self.stats = {"synthetic_test_only": True}
        self.close_count = 0
        self.video = self.targets = self.context_tokens = self.context_key = None

    def prepare(self, frames):
        values = np.asarray([frame[0, 0, 0] for frame in frames], dtype=np.float64)
        self.targets = np.repeat(values.reshape(-1, 2).mean(axis=1), self.grid ** 2)
        return len(values) // 2

    def predict_errors(self, context_ids, target_ids):
        if np.intersect1d(context_ids, target_ids).size:
            raise AssertionError("context/target overlap")
        return self.targets[target_ids] + 1.0

    def close(self):
        self.close_count += 1


class FakeIBackend:
    """Contract shim for the PRIMARY seed-bound collator fix, not worker code."""
    grid = 2
    def __init__(self):
        self.stats = {"synthetic_test_only": True}
        self.seeds, self.mask_history = [], []
        lock = threading.Lock()
        self.collator = SimpleNamespace(_itr_counter=SimpleNamespace(value=123, get_lock=lambda: lock))
        self.close_count = 0

    def batch_errors(self, frames, seed):
        self.seeds.append(seed)
        # Exercise the primary counter context itself, with a NumPy collator shim.
        with jepa.seeded_mask_counter(self.collator, seed):
            self.collator._itr_counter.value += 1  # Upstream collator.step().
            rng = np.random.default_rng(self.collator._itr_counter.value)
            masks = np.tile(rng.choice(4, size=2, replace=False)[None, None, :], (2, len(frames), 1))
            values = np.asarray([float(frame[0, 0, 0]) for frame in frames])
            errors = values[None, :, None] + masks + 1.0
            self.mask_history.append(masks.copy())
        return masks, errors

    def close(self):
        self.close_count += 1


class FakeAdapters(worker.LocalAdapters):
    def __init__(self, root):
        super().__init__()
        self.root, self.active = root, None
        self.loads = {kind: 0 for kind in ("rgb", "vjepa", "ijepa")}
        self.closes = {kind: 0 for kind in self.loads}
        self.scopes, self.run_kinds, self.sample_configs = [], [], []
        self.fail_model, self.empty_model, self.fail_load = None, None, None
        self.probe_delta = 0
        self.checkpoints = {}
        for kind in self.loads:
            path = root / (kind + ".fake-weights")
            path.write_bytes(("synthetic weights " + kind).encode())
            self.checkpoints[kind] = path

    def validate_profiles(self, profiles, channels):
        for channel in channels:
            if not isinstance(profiles.get(channel), dict):
                raise worker.WorkerError("missing_bundle_profile", "missing " + channel, 422)
        if "corrected" in channels and profiles["corrected"]["config"] != asdict(jepa.Config(bidirectional=True)):
            raise worker.WorkerError("corrected_profile_mismatch", "fake test expects exact config", 422)

    def probe(self, video):
        return {"total_frames": (40 if video.name == "b.mp4" else 32) + self.probe_delta,
                "fps": 24.0 if video.name == "b.mp4" else 10.0}

    def model_files(self, model, profile):
        return {"checkpoint": self.checkpoints[model]}

    def check_model_claims(self, model, identity, profile):
        pass

    def load_rgb(self, profile):
        self.loads["rgb"] += 1
        if self.fail_load == "rgb":
            raise RuntimeError("synthetic cold-load failure")
        return {"synthetic_test_only": True}

    def check_loaded_rgb(self, runtime, profile):
        pass

    def extract_rgb(self, runtime, video, target, profile, probe, events):
        if self.fail_model == "rgb":
            events.append({"stage": "inference_batch", "error_type": "RuntimeError"})
            raise RuntimeError("synthetic RGB forward failure")
        with target.open("xb") as handle:
            np.savez_compressed(handle, features=np.full((probe["total_frames"], 512), 3, np.float32))
        return {"frame_count": probe["total_frames"], "fps": probe["fps"], "feature_shape": [probe["total_frames"], 512]}

    def sample_corrected(self, video, config):
        self.sample_configs.append(asdict(config))
        count = 40 if video.name == "b.mp4" else 32
        fps = 24.0 if video.name == "b.mp4" else 10.0
        ids = jepa.uniform_frame_ids(count, config.max_frames)
        keys = jepa.uniform_frame_ids(count, config.max_keyframes)
        offset = 64 if video.name == "b.mp4" else 0
        make = lambda fids: [np.full((2, 2, 3), int(fid) + offset, np.uint8) for fid in fids]
        return jepa.SampledVideo(make(ids), ids, make(keys), keys, count, fps,
                                {"status": "ok", "synthetic_test_only": True})

    @contextmanager
    def isolation(self, kind):
        if self.active is not None:
            raise AssertionError("overlapping isolation")
        self.active = kind
        self.scopes.append(("enter", kind))
        try:
            yield SimpleNamespace(kind=kind)
        finally:
            self.scopes.append(("exit", kind))
            self.active = None

    def make_backend(self, kind, legacy, config, identity):
        assert self.active == kind
        self.loads[kind] += 1
        if self.fail_load == kind:
            raise RuntimeError("synthetic backend-load failure")
        return FakeVBackend() if kind == "vjepa" else FakeIBackend()

    def run_backend(self, kind, backend, sampled, config):
        assert self.active == kind
        self.run_kinds.append(kind)
        if self.fail_model == kind:
            raise RuntimeError("synthetic JEPA forward failure")
        if self.empty_model == kind:
            return jepa.empty_evidence(kind, len(sampled.frames) // 2 if kind == "vjepa" else len(sampled.keyframes), "synthetic missing")
        return super().run_backend(kind, backend, sampled, config)

    def close_rgb(self, runtime):
        self.closes["rgb"] += 1

    def close_backend(self, kind, backend):
        assert self.active == kind
        self.closes[kind] += 1
        backend.close()


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="worker-cpu-test-")
        self.root = Path(self.temp.name)
        self.output_root, self.videos = self.root / "output", self.root / "videos"
        self.output_root.mkdir()
        self.videos.mkdir()
        for name in ("a.mp4", "b.mp4"):
            (self.videos / name).write_bytes(("synthetic input " + name).encode())
        self.bundle = self.output_root / "bundle.json"
        self.profiles = {"rgb": rgb_profile(), "corrected": corrected_profile()}
        self.write_bundle()
        self.adapters = FakeAdapters(self.root)
        self.policy = worker.PathPolicy(self.output_root, (self.output_root, self.videos), self.output_root)
        self.core = worker.FeatureWorker(self.adapters, self.policy)

    def tearDown(self):
        self.core.close()
        self.temp.cleanup()

    def write_bundle(self, value=None):
        self.bundle.write_text(json.dumps({"feature_profiles": self.profiles} if value is None else value), encoding="utf-8")

    def request(self, directory="request-1", channels=None, video="a.mp4"):
        result = {"sourcevideo": str(self.videos / video), "output": str(self.output_root / directory),
                  "profile_json": str(self.bundle)}
        if channels is not None:
            result["channels"] = channels
        return result

    def run_request(self, **kwargs):
        return self.core.process(self.request(**kwargs), http=True)


class CoreTests(Fixture):
    def test_import_and_health_are_cpu_only_and_lazy(self):
        imported_before=set(sys.modules)
        health = self.core.health()
        self.assertEqual(health["schema"], worker.SCHEMA)
        self.assertEqual(health["processing"], "serial")
        self.assertEqual(health["hot_models"], [])
        self.assertRegex(health["worker_code_sha256"], r"^[0-9a-f]{64}$")
        self.assertFalse({"torch","cv2"} & (set(sys.modules)-imported_before))
        # Other suites may have loaded these packages already. Check import +
        # production health laziness in a genuinely fresh interpreter as well.
        import subprocess
        code="import sys; sys.path.insert(0, "+repr(str(Path(worker.__file__).parent))+"); import optimized_model_worker as w; core=w.FeatureWorker(); core.health(); assert 'torch' not in sys.modules; assert 'cv2' not in sys.modules"
        probe=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True)
        self.assertEqual(probe.returncode,0,probe.stderr)
        self.assertEqual(sum(self.adapters.loads.values()), 0)

    def test_client_three_fields_and_exact_features_interface(self):
        status, result = self.run_request()
        self.assertEqual(status, 200, result)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(set(result["features"]), {"rgb", "corrected"})
        self.assertEqual(result["total_frames"], 32)
        self.assertEqual(result["fps"], 10.0)
        for channel in result["features"]:
            entry = result["features"][channel]
            self.assertTrue(Path(entry["npz_path"]).is_absolute())
            self.assertTrue(Path(entry["npz_path"]).is_file())
            self.assertEqual(entry["fps"], result["fps"])
            self.assertEqual(entry["total_frames"], result["total_frames"])
        self.assertEqual(result["features"]["rgb"]["feature_key"], "features")
        self.assertEqual(result["features"]["corrected"]["feature_key"], "signals")
        self.assertEqual(result["features"]["corrected"]["names_key"], "names")
        self.assertEqual(result["features"]["corrected"]["shape"], [32, 14])
        with np.load(result["features"]["corrected"]["npz_path"], allow_pickle=False) as data:
            self.assertEqual(set(data.files), {"signals", "names", "fps", "frame_ids", "frame_count", "metadata_json"})
            self.assertEqual(data["signals"].dtype, np.float32)
            self.assertEqual(data["names"].tolist(), worker.CORRECTED_NAMES)
            self.assertTrue(np.isfinite(data["signals"]).all())
        self.assertEqual(result["model_load_status"], {kind: "cold_loaded" for kind in ("rgb", "vjepa", "ijepa")})
        self.assertFalse(result["feature_cache_used"])
        self.assertEqual(self.adapters.closes, {kind: 0 for kind in self.adapters.closes})

    def test_only_requested_rgb_loads_no_jepa(self):
        status, result = self.run_request(channels=["rgb"])
        self.assertEqual(status, 200, result)
        self.assertEqual(set(result["features"]), {"rgb"})
        self.assertEqual(self.adapters.loads, {"rgb": 1, "vjepa": 0, "ijepa": 0})
        self.assertFalse((self.output_root / "request-1" / "raw").exists())

    def test_only_corrected_has_no_rgb_load(self):
        status, result = self.run_request(channels="corrected")
        self.assertEqual(status, 200, result)
        self.assertEqual(set(result["features"]), {"corrected"})
        self.assertEqual(self.adapters.loads, {"rgb": 0, "vjepa": 1, "ijepa": 1})
        self.assertEqual(self.adapters.sample_configs, [self.profiles["corrected"]["config"]])

    def test_hot_cache_request_sequence_independent_masks_and_features(self):
        status, first = self.run_request(directory="a-cold", channels=["corrected"])
        self.assertEqual(status, 200, first)
        backend = self.core.models["ijepa"].resource
        initial_masks = [value.copy() for value in backend.mask_history]
        status, middle = self.run_request(directory="b-hot", channels=["corrected"], video="b.mp4")
        self.assertEqual(status, 200, middle)
        status, repeated = self.run_request(directory="a-hot", channels=["corrected"])
        self.assertEqual(status, 200, repeated)
        self.assertIs(self.core.models["ijepa"].resource, backend)
        self.assertEqual(self.adapters.loads, {"rgb": 0, "vjepa": 1, "ijepa": 1})
        self.assertEqual(backend.seeds, [0, 2, 4, 6] * 3)
        self.assertEqual(backend.collator._itr_counter.value, 123)
        for before, after in zip(initial_masks, backend.mask_history[-4:]):
            np.testing.assert_array_equal(before, after)
        with np.load(first["features"]["corrected"]["npz_path"]) as a, np.load(repeated["features"]["corrected"]["npz_path"]) as b:
            np.testing.assert_array_equal(a["signals"], b["signals"])
        for key in ("ijepa_patch_counts", "ijepa_raw_errors", "ijepa_raw_heatmaps", "vjepa_patch_counts", "vjepa_raw_errors"):
            with np.load(first["paths"]["raw_npz"]) as a, np.load(repeated["paths"]["raw_npz"]) as b:
                np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(repeated["model_load_status"], {"vjepa": "hot_reused", "ijepa": "hot_reused"})
        self.assertEqual(self.adapters.scopes, [(action, kind) for _ in range(3) for kind in ("vjepa", "ijepa") for action in ("enter", "exit")])
        self.assertIsNone(self.adapters.active)
        self.assertEqual(self.adapters.closes, {kind: 0 for kind in self.adapters.closes})

    def test_hot_rgb_weight_hash_is_not_repeated(self):
        original = worker.sha256_file
        calls = []
        def traced(path):
            calls.append(Path(path))
            return original(path)
        with patch.object(worker, "sha256_file", traced):
            self.assertEqual(self.run_request(directory="cold", channels=["rgb"])[0], 200)
            status, result = self.run_request(directory="hot", channels=["rgb"])
        self.assertEqual(status, 200, result)
        self.assertEqual(result["model_load_status"], {"rgb": "hot_reused"})
        self.assertEqual(self.adapters.loads["rgb"], 1)
        self.assertEqual(calls.count(self.adapters.checkpoints["rgb"]), 1)

    def test_close_only_at_shutdown_once_with_separate_isolations(self):
        self.assertEqual(self.run_request()[0], 200)
        self.assertEqual(self.core.close(), [])
        self.assertEqual(self.core.close(), [])
        self.assertEqual(self.adapters.closes, {kind: 1 for kind in self.adapters.closes})
        self.assertEqual(self.core.health()["hot_models"], [])
        self.assertEqual(self.core.health()["status"], "closed")
        self.assertIsNone(self.adapters.active)
        self.assertEqual(self.run_request(directory="closed")[0], 503)

    def test_missing_profile_is_error_before_loading_and_writing(self):
        self.write_bundle({"recipe": "no profiles"})
        status, result = self.run_request()
        self.assertEqual(status, 422)
        self.assertEqual(result["error"]["code"], "missing_bundle_profile")
        self.assertFalse((self.output_root / "request-1").exists())
        self.assertEqual(sum(self.adapters.loads.values()), 0)

    def test_unrequested_missing_profile_is_not_loaded(self):
        self.write_bundle({"feature_profiles": {"rgb": self.profiles["rgb"]}})
        self.assertEqual(self.run_request(channels=["rgb"])[0], 200)

    def test_load_and_forward_errors_never_become_normal_or_cache_hits(self):
        self.adapters.fail_load = "rgb"
        status, cold = self.run_request(directory="load-error", channels=["rgb"])
        self.assertEqual(status, 500)
        self.assertEqual(cold["status"], "error")
        self.assertEqual(cold["features"], {})
        self.assertEqual(cold["model_load_status"], {"rgb": "error"})
        self.assertTrue(Path(cold["paths"]["error"]).is_file())
        self.adapters.fail_load, self.adapters.fail_model = None, "ijepa"
        status, failed = self.run_request(directory="forward-error", channels=["corrected"])
        self.assertEqual(status, 500)
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["features"], {})
        self.assertEqual(failed["model_load_status"]["ijepa"], "error")
        self.assertNotIn("ijepa", self.core.health()["hot_models"])
        self.assertIn("ijepa", self.core.health()["failed_models"])
        self.assertIsNone(self.adapters.active)
        status, blocked = self.run_request(directory="quarantined", channels=["corrected"])
        self.assertEqual(status, 503)
        self.assertEqual(blocked["error"]["code"], "model_quarantined")
        self.assertFalse(blocked["feature_cache_used"])

    def test_no_evidence_is_not_an_all_zero_normal_feature(self):
        self.adapters.empty_model = "ijepa"
        status, result = self.run_request(channels=["corrected"])
        self.assertEqual(status, 500)
        self.assertEqual(result["error"]["code"], "missing_model_evidence")
        self.assertEqual(result["features"], {})

    def test_mismatched_frame_fps_is_error_not_truncation(self):
        self.adapters.probe_delta = 1
        status, result = self.run_request(channels=["corrected"])
        self.assertEqual(status, 422)
        self.assertEqual(result["error"]["code"], "frame_fps_mismatch")
        self.assertEqual(sum(self.adapters.loads.values()), 0)

    def test_checkpoint_changed_while_hot_requires_restart(self):
        self.assertEqual(self.run_request(directory="first", channels=["rgb"])[0], 200)
        self.adapters.checkpoints["rgb"].write_bytes(b"changed fake checkpoint")
        status, result = self.run_request(directory="second", channels=["rgb"])
        self.assertEqual(status, 409)
        self.assertEqual(result["error"]["code"], "hot_checkpoint_changed")
        self.assertEqual(self.adapters.loads["rgb"], 1)

    def test_core_serializes_concurrent_calls(self):
        original = self.adapters.extract_rgb
        active = [0, 0]
        guard = threading.Lock()
        def slow(*args):
            with guard:
                active[0] += 1
                active[1] = max(active[1], active[0])
            time.sleep(0.02)
            try:
                return original(*args)
            finally:
                with guard:
                    active[0] -= 1
        self.adapters.extract_rgb = slow
        results = []
        threads = [threading.Thread(target=lambda i=i: results.append(self.run_request(directory="serial-" + str(i), channels=["rgb"]))) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual([status for status, _ in results], [200, 200])
        self.assertEqual(active[1], 1)
        self.assertEqual(self.adapters.loads["rgb"], 1)


    def test_cli_writes_complete_worker_result_and_compact_stdout(self):
        output = self.output_root / "cli-complete"
        stream = io.StringIO()
        with patch.object(worker, "FeatureWorker", return_value=self.core), redirect_stdout(stream):
            code = worker.main(["--video", str(self.videos / "a.mp4"), "--output", str(output),
                                "--channels", "corrected", "--profile-json", str(self.bundle)])
        self.assertEqual(code, 0)
        compact = json.loads(stream.getvalue())
        full = json.loads((output / "worker_result.json").read_text(encoding="utf-8"))
        self.assertEqual(full["status"], "ok")
        self.assertEqual(compact["features"], full["features"])
        self.assertIn("identity", full)
        self.assertNotIn("identity", compact)
        self.assertEqual(full["paths"]["result"], str(output / "worker_result.json"))
        self.assertEqual(self.adapters.closes, {"rgb": 0, "vjepa": 1, "ijepa": 1})


    def test_cli_profile_rejection_still_saves_complete_new_output_result(self):
        self.write_bundle({"recipe": "missing required profiles"})
        output = self.output_root / "cli-profile-error"
        stream = io.StringIO()
        with patch.object(worker, "FeatureWorker", return_value=self.core), redirect_stdout(stream):
            code = worker.main(["--video", str(self.videos / "a.mp4"), "--output", str(output),
                                "--channels", "corrected", "--profile-json", str(self.bundle)])
        self.assertEqual(code, 1)
        full = json.loads((output / "worker_result.json").read_text(encoding="utf-8"))
        self.assertEqual(full["status"], "error")
        self.assertEqual(full["error"]["code"], "missing_bundle_profile")
        self.assertEqual(sum(self.adapters.loads.values()), 0)


class ValidationTests(Fixture):
    def test_motion_unknown_duplicate_empty_channels_rejected(self):
        for channels in (["motion"], ["rgb", "rgb"], [], "", ["jepa"], [1]):
            status, result = self.core.process(self.request(channels=channels), http=True)
            self.assertEqual(status, 400, result)
            self.assertEqual(result["error"]["code"], "invalid_channels")
        self.assertEqual(sum(self.adapters.loads.values()), 0)

    def test_video_alias_and_conflict(self):
        request = self.request(channels=["rgb"])
        request["video"] = request.pop("sourcevideo")
        self.assertEqual(self.core.process(request, http=True)[0], 200)
        request = self.request(directory="conflict", channels=["rgb"])
        request["video"] = request["sourcevideo"]
        self.assertEqual(self.core.process(request, http=True)[0], 400)

    def test_relative_paths_unknown_fields_and_external_paths_rejected(self):
        for field, value in (("sourcevideo", "relative.mp4"), ("output", "relative"),
                             ("profile_json", str(self.root / "outside.json")),
                             ("sourcevideo", str(self.root / "outside.mp4")),
                             ("output", str(self.root / "outside-output"))):
            request = self.request(channels=["rgb"])
            request[field] = value
            self.assertIn(self.core.process(request, http=True)[0], (400, 403))
        request = {**self.request(channels=["rgb"]), "shell": "not supported"}
        self.assertEqual(self.core.process(request, http=True)[0], 400)
        self.assertEqual(sum(self.adapters.loads.values()), 0)

    def test_symlink_escape_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "escaped.mp4").write_bytes(b"fake")
        (self.videos / "link").symlink_to(outside, target_is_directory=True)
        request = self.request(channels=["rgb"])
        request["sourcevideo"] = str(self.videos / "link" / "escaped.mp4")
        self.assertEqual(self.core.process(request, http=True)[0], 403)
        (self.output_root / "link").symlink_to(outside, target_is_directory=True)
        request = self.request(channels=["rgb"])
        request["output"] = str(self.output_root / "link" / "new")
        self.assertEqual(self.core.process(request, http=True)[0], 403)

    def test_saved_raw_directory_is_not_reused_and_cli_only_relaxes_output(self):
        request = self.request(channels=["rgb"])
        output = Path(request["output"])
        (output / "raw").mkdir(parents=True)
        (output / "raw" / "signals.npz").write_bytes(b"historical")
        status, result = self.core.process(request, http=True)
        self.assertEqual(status, 409)
        self.assertEqual(result["error"]["code"], "output_exists")
        request["output"] = str(self.root / "cli-output")
        self.assertEqual(self.core.process(request, http=False)[0], 200)
        request = self.request(channels=["rgb"])
        request["sourcevideo"] = str(self.root / "outside.mp4")
        self.assertEqual(self.core.process(request, http=False)[0], 403)

    def test_strict_json_rejects_duplicates_nonfinite_invalid_utf8(self):
        for body in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'\xff', '{"x":1}'.encode("utf-16")):
            with self.assertRaises((ValueError, UnicodeError)):
                worker.strict_json(body)

    def test_real_rgb_profile_validator_and_tampering(self):
        adapter = worker.LocalAdapters()
        adapter.validate_profiles({"rgb": self.profiles["rgb"]}, ["rgb"])
        for key, value in (("profile_id", "0" * 64), ("fps_source", "guess FPS")):
            profile = {**self.profiles["rgb"], key: value}
            with self.assertRaises(worker.WorkerError):
                adapter.validate_profiles({"rgb": profile}, ["rgb"])
        profile = json.loads(json.dumps(self.profiles["rgb"]))
        profile["preprocess"]["mean"][0] += 0.1
        profile["profile_id"] = hashlib.sha256(worker.canonical_json({k:v for k,v in profile.items() if k!="profile_id"}).encode()).hexdigest()
        with self.assertRaisesRegex(worker.WorkerError, "preprocessing"):
            adapter.validate_profiles({"rgb": profile}, ["rgb"])

    def test_real_corrected_validator_requires_full_config_and_primary_counter_policy(self):
        adapter = worker.LocalAdapters()
        for key, value in (("seed", 1), ("bidirectional", 1), ("mask_passes", 2), ("max_frames", 64)):
            profile = json.loads(json.dumps(self.profiles["corrected"]))
            profile["config"][key] = value
            with self.assertRaises(worker.WorkerError):
                adapter.validate_profiles({"corrected": profile}, ["corrected"])
        profile = json.loads(json.dumps(self.profiles["corrected"]))
        del profile["config"]["ijepa_batch_size"]
        with self.assertRaisesRegex(worker.WorkerError, "every Config field"):
            adapter.validate_profiles({"corrected": profile}, ["corrected"])
        with patch.object(worker.inspect, "getsource", return_value="old CPU RNG-only implementation"):
            with self.assertRaises(worker.WorkerError) as raised:
                adapter.validate_profiles({"corrected": self.profiles["corrected"]}, ["corrected"])
            self.assertEqual(raised.exception.code, "ijepa_counter_policy_missing")
        with patch.object(worker.inspect, "getsource", return_value="primary seeded_mask_counter seed-bound _itr_counter implementation"):
            adapter.validate_profiles({"corrected": self.profiles["corrected"]}, ["corrected"])
        self.assertNotIn("manual_seed", Path(worker.__file__).read_text(encoding="utf-8"))

    def test_private_ijepa_factory_is_used_without_worker_seed_patch(self):
        adapter = worker.LocalAdapters()
        checkpoint = self.adapters.checkpoints["ijepa"]
        factory = unittest.mock.Mock(return_value=object())
        module = SimpleNamespace(_RealIJEPA=factory)
        legacy = SimpleNamespace(torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
                                 CKPT_SLIM_BF16=checkpoint, CKPT_FULL=checkpoint)
        identity = {"files": {"checkpoint": {"path": str(checkpoint.resolve())}}}
        with patch.object(adapter, "module", return_value=module):
            result = adapter.make_backend("ijepa", legacy, jepa.Config(bidirectional=True), identity)
        self.assertIs(result, factory.return_value)
        factory.assert_called_once_with(legacy, 0)


    def test_primary_v2_profile_v1_name_accepts_adapter_snapshot_provenance(self):
        profile = corrected_profile()
        profile["name"] = "corrected-true-jepa-absolute-patch-v1"
        profile["adapter_sha256"] = "b" * 64  # Historical whole-file snapshot, not a feature gate.
        worker.LocalAdapters().validate_profiles({"corrected": profile}, ["corrected"])
        profile.pop("i_mask_reuse_policy")
        with self.assertRaises(worker.WorkerError):
            worker.LocalAdapters().validate_profiles({"corrected": profile}, ["corrected"])

    def test_batch_cli_changes_do_not_change_descriptor_compatibility_identity(self):
        path = self.root / "adapter-copy.py"
        original = Path(worker.__file__).with_name("build_optimization_jepa.py").read_text(encoding="utf-8")
        path.write_text(original, encoding="utf-8")
        module = SimpleNamespace(__file__=str(path))
        first = worker.LocalAdapters._implementation_identity("build_optimization_jepa", module)
        path.write_text(original + "\n# Changed batch-only stage validation/provenance\n", encoding="utf-8")
        second = worker.LocalAdapters._implementation_identity("build_optimization_jepa", module)
        self.assertEqual(first, second)


class HTTPTests(Fixture):
    def setUp(self):
        super().setUp()
        self.server = worker.SerialHTTPServer(("127.0.0.1", 0), self.core)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(5)
        self.server.server_close()
        super().tearDown()

    def http(self, method="GET", path="/health", body=None, headers=None, include_length=True):
        conn = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        conn.putrequest(method, path, skip_host=True)
        headers = {"Host": "127.0.0.1:" + str(self.server.server_port), **(headers or {})}
        if body is not None and include_length:
            headers.setdefault("Content-Length", str(len(body)))
        for key, value in headers.items():
            conn.putheader(key, value)
        conn.endheaders(body)
        response = conn.getresponse()
        data = json.loads(response.read())
        status = response.status
        conn.close()
        return status, data

    def test_health_no_models_and_serial_http_server(self):
        status, response = self.http()
        self.assertEqual(status, 200)
        self.assertEqual(response["processing"], "serial")
        self.assertEqual(response["hot_models"], [])
        self.assertNotIsInstance(self.server, ThreadingHTTPServer)
        self.assertEqual(self.http(path="/missing")[0], 404)

    def test_infer_exact_client_request_and_features_json(self):
        body = json.dumps(self.request(channels=["rgb"])).encode()
        status, response = self.http("POST", "/infer", body, {"Content-Type": "application/json"})
        self.assertEqual(status, 200, response)
        self.assertTrue(Path(response["features"]["rgb"]["npz_path"]).is_file())
        self.assertEqual(self.http()[1]["hot_models"], ["rgb"])

    def test_nonlocal_host_or_origin_rejected_before_any_loading(self):
        for headers in ({"Host": "evil.example:" + str(self.server.server_port)},
                        {"Origin": "https://evil.example"}, {"Origin": "null"}):
            self.assertEqual(self.http(headers=headers)[0], 403)
        self.assertEqual(self.http(headers={"Origin": "http://localhost:5000"})[0], 200)
        self.assertEqual(sum(self.adapters.loads.values()), 0)

    def test_content_length_limit_transfer_encoding_content_type_and_json(self):
        base = {"Content-Type": "application/json"}
        self.assertEqual(self.http("POST", "/infer", headers=base)[0], 411)
        for headers, expected in (({**base, "Content-Length": "-1"}, 400),
                                  ({**base, "Content-Length": str(worker.MAX_BODY_BYTES + 1)}, 413),
                                  ({**base, "Content-Length": "999999999999999999999"}, 400),
                                  ({**base, "Transfer-Encoding": "chunked", "Content-Length": "2"}, 400)):
            self.assertEqual(self.http("POST", "/infer", headers=headers)[0], expected)
        self.assertEqual(self.http("POST", "/infer", b"{}")[0], 415)
        for body in (b"{", b'{"x":NaN}', b'{"x":1,"x":2}', b"\xff"):
            self.assertEqual(self.http("POST", "/infer", body, base)[0], 400)
        self.assertEqual(sum(self.adapters.loads.values()), 0)

    def test_duplicate_host_and_length_headers_rejected(self):
        for duplicate in ("Host", "Content-Length"):
            conn = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
            conn.putrequest("POST", "/infer", skip_host=True)
            conn.putheader("Host", "127.0.0.1:" + str(self.server.server_port))
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", "2")
            conn.putheader(duplicate, "2" if duplicate == "Content-Length" else "localhost:" + str(self.server.server_port))
            conn.endheaders(b"{}")
            response = conn.getresponse()
            self.assertIn(response.status, (400, 403))
            response.read()
            conn.close()


if __name__ == "__main__":
    unittest.main()
