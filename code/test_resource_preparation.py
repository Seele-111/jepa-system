"""Bounded prepare-tool regression tests: tiny local ZIPs and fake Torch only.

Safety assertions deliberately fail if a rejected partial source tree is reused
or conversion overwrites a late-created external hardlink. No real registry,
checkpoint, video, GPU, WSL process, HTTP server, or network transfer is used.
"""
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path, PureWindowsPath
import stat
import sys
import tempfile
import tomllib
import types
import unittest
from unittest.mock import Mock, patch
import warnings
import zipfile

import prepare_model_resources as prepare
from published_models import source_tree_digest


class ResourcePreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="jepa-prepare-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.directory = self.root / "resources"
        self.directory.mkdir()
        self.outside = self.root / "outside-resources"
        self.outside.mkdir()
        self.sentinel = self.outside / "protected.pt"
        self.original = b"external file must remain unchanged"
        self.sentinel.write_bytes(self.original)
        self.sources = {
            "src/models/tiny.py": b"VALUE = 1\n",
            "app/infer.py": b"def infer():\n    return 1\n",
        }
        reference = self.root / "reviewed-source"
        for name, data in {**self.sources, "LICENSE": b"fixture license\n"}.items():
            path = reference / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.reference = reference
        source_claim = {
            "source_tree_sha256": source_tree_digest(reference),
            "repository": "https://fixtures.invalid/upstream",
            "revision": "test-reviewed-revision",
        }
        self.artifacts = {
            "vjepa_predictor": b"fixture vjepa full checkpoint",
            "ijepa_full": b"fixture ijepa full checkpoint",
            "r3d_checkpoint": b"fixture r3d checkpoint",
            "vjepa_encoder": b"fixture converted vjepa",
            "ijepa_checkpoint": b"fixture converted ijepa",
        }
        self.claims = {kind + "_source": copy.deepcopy(source_claim)
                       for kind in ("vjepa", "ijepa")}
        for key, data in self.artifacts.items():
            self.claims[key] = {"sha256": self.sha(data),
                                "source_url": "https://fixtures.invalid/" + key}
        self.conversion = {"torch_version": "test-pinned-torch",
                           "vjepa_encoder_branch": "target_encoder",
                           "ijepa_branches": ["encoder", "target_encoder", "predictor"]}
        self.release = {"resources": self.claims, "conversion": self.conversion}
        self.enterContext(patch.object(prepare, "registry", return_value=self.release))
        self.urlopen = self.enterContext(patch.object(
            prepare, "urlopen", side_effect=AssertionError("real network access is forbidden")))
        self.torch = types.ModuleType("torch")
        self.torch.__version__ = self.conversion["torch_version"]
        self.torch.bfloat16 = object()
        self.torch.load = Mock(return_value={"target_encoder": {"module.backbone.layer": 1}})
        self.torch.save = Mock(side_effect=lambda state, path: Path(path).write_bytes(
            self.artifacts["vjepa_encoder"]))
        self.enterContext(patch.dict(sys.modules, {"torch": self.torch}))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    @staticmethod
    def sha(data):
        return hashlib.sha256(data).hexdigest()

    def source_entries(self, license=True):
        entries = [("upstream/" + name, data) for name, data in self.sources.items()]
        if license:
            entries.append(("upstream/LICENSE", b"fixture license\n"))
        return entries

    def archive_bytes(self, entries):
        buffer = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(buffer, "w") as bundle:
                for name, data in entries:
                    bundle.writestr(name, data)
        return buffer.getvalue()

    def install_archive(self, entries=None, kind="vjepa", directory=None):
        directory = self.directory if directory is None else directory
        name = "vjepa2" if kind == "vjepa" else "ijepa"
        archive = directory / "downloads" / (name + "-test-reviewed-revision.zip")
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(self.archive_bytes(self.source_entries() if entries is None else entries))
        return archive

    def symlink_entry(self):
        entry = zipfile.ZipInfo("upstream/src/external_link.py")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        return entry, str(self.sentinel).encode()

    def late_hardlink(self, target):
        try:
            os.link(self.sentinel, target)
        except (OSError, NotImplementedError) as exc:
            self.skipTest("temporary filesystem does not support hardlinks: " + str(exc))

    def full_args(self, config=None, transport="native"):
        for kind in ("vjepa", "ijepa"):
            self.install_archive(kind=kind)
        paths = {}
        for key, data in self.artifacts.items():
            path = self.directory / "checkpoints" / {
                "vjepa_encoder": "encoder_only.pt",
                "ijepa_checkpoint": "ijepa_true_slim_bf16.pt",
            }.get(key, key + ".pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            paths[key] = path
        args = ["--directory", str(self.directory), "--accept-noncommercial",
                "--vjepa-full", str(paths["vjepa_predictor"]),
                "--ijepa-full", str(paths["ijepa_full"]),
                "--r3d", str(paths["r3d_checkpoint"]), "--transport", transport]
        if config is not None:
            args += ["--write-config", str(config)]
        return args

    def test_sources_only_downloads_tiny_mock_archives_not_checkpoints(self):
        payload = self.archive_bytes(self.source_entries())
        self.urlopen.side_effect = lambda *args, **kwargs: io.BytesIO(payload)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(prepare.main(["--directory", str(self.directory),
                                          "--sources-only", "--accept-noncommercial"]), 0)
        result = json.loads(output.getvalue().splitlines()[-1])
        self.assertEqual(result["status"], "sources_verified")
        self.assertEqual(self.urlopen.call_count, 2)
        self.assertFalse((self.directory / "checkpoints").exists())
        self.torch.load.assert_not_called()
        self.torch.save.assert_not_called()

    def test_source_filter_keeps_inference_notices_and_omits_training_case_collisions(self):
        entries = self.source_entries() + [
            ("upstream/NOTICE", b"fixture notice"),
            ("upstream/README.md", b"fixture readme"),
            ("upstream/configs/train.yaml", b"lower"),
            ("upstream/configs/TRAIN.yaml", b"upper"),
            ("upstream/weights/large.pt", b"not a real checkpoint"),
        ]
        self.install_archive(entries)
        target = prepare.prepare_source("vjepa", self.directory, self.claims)
        self.assertEqual(source_tree_digest(target), self.claims["vjepa_source"]["source_tree_sha256"])
        self.assertTrue((target / "LICENSE").is_file())
        self.assertEqual((target / "NOTICE").read_bytes(), b"fixture notice")
        self.assertFalse((target / "configs").exists())
        self.assertFalse((target / "weights").exists())
        self.urlopen.assert_not_called()

    def test_verified_existing_source_is_reused_without_network(self):
        self.install_archive()
        target = prepare.prepare_source("vjepa", self.directory, self.claims)
        self.assertEqual(prepare.prepare_source("vjepa", self.directory, self.claims), target)
        self.assertEqual((target / "src/models/tiny.py").read_bytes(), self.sources["src/models/tiny.py"])
        self.urlopen.assert_not_called()

    def test_existing_modified_or_incomplete_source_is_not_overwritten(self):
        target = self.directory / "vjepa2"
        path = target / "src/models/tiny.py"
        path.parent.mkdir(parents=True)
        for data in (b"modified\n", self.sources["src/models/tiny.py"]):
            with self.subTest(data=data):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    prepare.prepare_source("vjepa", self.directory, self.claims)
                self.assertEqual(path.read_bytes(), data)
        self.urlopen.assert_not_called()

    def test_path_traversal_absolute_and_backslash_members_cannot_escape(self):
        names = ["upstream/src/../../../outside-resources/protected.pt",
                 "upstream/src/../outside.py", "upstream\\src\\outside.py",
                 "upstream/src/..\\outside.py", "/upstream/src/outside.py",
                 self.sentinel.as_posix(), "C:/outside/protected.pt"]
        for index, name in enumerate(names):
            with self.subTest(name=name):
                directory = self.root / ("bad-path-" + str(index))
                self.install_archive(self.source_entries() + [(name, b"overwrite")], directory=directory)
                with self.assertRaises((ValueError, OSError)):
                    prepare.prepare_source("vjepa", directory, self.claims)
                self.assertEqual(self.sentinel.read_bytes(), self.original)
                self.assertFalse((directory / "outside.py").exists())

    def test_zip_symlink_is_rejected_without_touching_external_file(self):
        self.install_archive(self.source_entries() + [self.symlink_entry()])
        with self.assertRaisesRegex(ValueError, "unsafe link"):
            prepare.prepare_source("vjepa", self.directory, self.claims)
        self.assertEqual(self.sentinel.read_bytes(), self.original)
        self.assertFalse((self.directory / "vjepa2/src/external_link.py").exists())

    def test_duplicate_inference_member_cannot_overwrite_first_file(self):
        name = "src/models/tiny.py"
        self.install_archive(self.source_entries() + [("upstream/" + name, b"overwrite")])
        with self.assertRaises((ValueError, FileExistsError)):
            prepare.prepare_source("vjepa", self.directory, self.claims)
        first = self.directory / "vjepa2" / name
        if first.exists():
            self.assertEqual(first.read_bytes(), self.sources[name])

    def test_case_colliding_inference_member_is_not_silently_accepted_or_overwritten(self):
        self.install_archive(self.source_entries() + [("upstream/src/models/TINY.py", b"overwrite")])
        with self.assertRaises((ValueError, FileExistsError)):
            prepare.prepare_source("vjepa", self.directory, self.claims)
        first = self.directory / "vjepa2/src/models/tiny.py"
        if first.exists():
            self.assertEqual(first.read_bytes(), self.sources["src/models/tiny.py"])

    def test_multiple_archive_roots_are_rejected_before_creating_source_target(self):
        self.install_archive(self.source_entries() + [("other/src/extra.py", b"extra")])
        with self.assertRaisesRegex(ValueError, "layout"):
            prepare.prepare_source("vjepa", self.directory, self.claims)
        self.assertFalse((self.directory / "vjepa2").exists())

    def test_unpinned_source_bytes_are_rejected(self):
        entries = self.source_entries()
        entries[0] = (entries[0][0], b"VALUE = 2\n")
        self.install_archive(entries)
        before = copy.deepcopy(self.release)
        with self.assertRaises(ValueError):
            prepare.prepare_source("vjepa", self.directory, self.claims)
        self.assertEqual(self.release, before)

    def test_missing_license_failure_cannot_be_bypassed_by_retry(self):
        self.install_archive(self.source_entries(license=False))
        for attempt in ("initial extraction", "retry after rejection"):
            with self.subTest(attempt=attempt):
                with self.assertRaises(ValueError):
                    prepare.prepare_source("vjepa", self.directory, self.claims)

    def test_unsafe_member_failure_cannot_leave_a_reusable_partial_source(self):
        self.install_archive(self.source_entries() + [self.symlink_entry()])
        for attempt in ("initial extraction", "retry after rejection"):
            with self.subTest(attempt=attempt):
                with self.assertRaises(ValueError):
                    prepare.prepare_source("vjepa", self.directory, self.claims)
                self.assertEqual(self.sentinel.read_bytes(), self.original)

    def test_download_verifies_mock_response_hash(self):
        payload = b"tiny download"
        self.urlopen.side_effect = None
        self.urlopen.return_value = io.BytesIO(payload)
        target = self.directory / "downloads/checkpoint.pt"
        self.assertEqual(prepare.download("https://fixtures.invalid/file", target, self.sha(payload)), target)
        self.assertEqual(target.read_bytes(), payload)
        self.urlopen.assert_called_once_with("https://fixtures.invalid/file", timeout=120)

    def test_matching_existing_download_is_reused_without_network(self):
        target = self.directory / "existing.pt"
        payload = b"matching fixture"
        target.write_bytes(payload)
        self.assertEqual(prepare.download("unused", target, self.sha(payload)), target)
        self.urlopen.assert_not_called()

    def test_existing_download_is_never_overwritten_without_matching_claim(self):
        for expected in (None, self.sha(b"different")):
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError):
                    prepare.download("unused", self.sentinel, expected)
                self.assertEqual(self.sentinel.read_bytes(), self.original)
        self.urlopen.assert_not_called()

    def test_rejected_download_is_kept_for_diagnosis_but_not_reused_as_verified(self):
        self.urlopen.side_effect = None
        self.urlopen.return_value = io.BytesIO(b"wrong bytes")
        target = self.directory / "download.pt"
        expected = self.sha(b"reviewed bytes")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            prepare.download("https://fixtures.invalid/file", target, expected)
        self.assertEqual(target.read_bytes(), b"wrong bytes")
        with self.assertRaisesRegex(ValueError, "not overwritten"):
            prepare.download("https://fixtures.invalid/file", target, expected)
        self.assertEqual(self.urlopen.call_count, 1)

    def test_interrupted_source_archive_is_not_trusted_on_retry(self):
        class Interrupted(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise OSError("fixture connection interrupted")
                return super().read(size)
        self.urlopen.side_effect = None
        self.urlopen.return_value = Interrupted(b"partial ZIP header")
        with self.assertRaises(OSError):
            prepare.prepare_source("vjepa", self.directory, self.claims)
        with self.assertRaises(zipfile.BadZipFile):
            prepare.prepare_source("vjepa", self.directory, self.claims)
        self.assertEqual(self.urlopen.call_count, 1)
        self.assertFalse((self.directory / "vjepa2").exists())

    def test_download_exclusive_open_preserves_late_external_hardlink(self):
        target = self.directory / "late-download.pt"
        def response(*args, **kwargs):
            self.late_hardlink(target)
            return io.BytesIO(b"replacement")
        self.urlopen.side_effect = response
        with self.assertRaises(FileExistsError):
            prepare.download("https://fixtures.invalid/file", target, self.sha(b"replacement"))
        self.assertEqual(self.sentinel.read_bytes(), self.original)

    def test_existing_matching_conversion_skips_torch(self):
        target = self.directory / "encoder_only.pt"
        target.write_bytes(self.artifacts["vjepa_encoder"])
        self.assertEqual(prepare.convert_checkpoint("unused", target, "vjepa", self.claims,
                                                    self.conversion), target.resolve())
        self.torch.load.assert_not_called()
        self.torch.save.assert_not_called()

    def test_existing_wrong_conversion_cannot_be_overwritten(self):
        target = self.directory / "encoder_only.pt"
        target.write_bytes(b"wrong existing checkpoint")
        with self.assertRaises(ValueError):
            prepare.convert_checkpoint("unused", target, "vjepa", self.claims, self.conversion)
        self.assertEqual(target.read_bytes(), b"wrong existing checkpoint")
        self.torch.load.assert_not_called()
        self.torch.save.assert_not_called()

    def test_conversion_requires_pinned_torch_before_loading_any_checkpoint(self):
        self.torch.__version__ = "different-torch-version"
        with self.assertRaisesRegex(ValueError, "pinned Torch"):
            prepare.convert_checkpoint("unused", self.directory / "new.pt", "vjepa",
                                       self.claims, self.conversion)
        self.torch.load.assert_not_called()
        self.torch.save.assert_not_called()

    def test_vjepa_conversion_uses_cpu_weights_only_mmap_and_verified_bytes(self):
        target = self.directory / "encoder_only.pt"
        result = prepare.convert_checkpoint("fake-full.pt", target, "vjepa", self.claims,
                                            self.conversion)
        self.assertEqual(result, target.resolve())
        self.torch.load.assert_called_once_with("fake-full.pt", weights_only=True, mmap=True,
                                                map_location="cpu")
        self.torch.save.assert_called_once()
        saved_state, staged = self.torch.save.call_args.args
        self.assertEqual(saved_state, {"encoder": {"layer": 1}})
        self.assertEqual(staged.name, target.name)
        self.assertNotEqual(staged, target)
        self.assertTrue(staged.parent.is_relative_to(self.directory))
        self.assertFalse(staged.parent.exists(), "owned conversion staging must be cleaned up")
        self.assertEqual(self.sha(target.read_bytes()), self.claims["vjepa_encoder"]["sha256"])

    def test_ijepa_conversion_keeps_all_reviewed_branches_and_casts_to_bfloat16(self):
        tensors = {branch: Mock() for branch in self.conversion["ijepa_branches"]}
        self.torch.load.return_value = {branch: {"module.layer": tensor}
                                       for branch, tensor in tensors.items()}
        self.torch.save.side_effect = lambda state, path: Path(path).write_bytes(
            self.artifacts["ijepa_checkpoint"])
        target = self.directory / "ijepa_true_slim_bf16.pt"
        prepare.convert_checkpoint("fake-full.pt", target, "ijepa", self.claims, self.conversion)
        state = self.torch.save.call_args.args[0]
        self.assertEqual(set(state), set(tensors))
        for branch, tensor in tensors.items():
            tensor.to.assert_called_once_with(self.torch.bfloat16)
            self.assertIs(state[branch]["layer"], tensor.to.return_value)
        self.torch.load.assert_called_once_with("fake-full.pt", weights_only=True, mmap=True,
                                                map_location="cpu")

    def test_conversion_output_hash_cannot_be_fixed_by_mutating_claims(self):
        before = copy.deepcopy(self.release)
        self.torch.save.side_effect = lambda state, path: Path(path).write_bytes(b"wrong conversion")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            prepare.convert_checkpoint("fake-full.pt", self.directory / "new.pt", "vjepa",
                                       self.claims, self.conversion)
        self.assertEqual(self.release, before)

    def test_conversion_preserves_late_external_hardlink_instead_of_overwriting(self):
        target = self.directory / "encoder_only.pt"
        def load(*args, **kwargs):
            # A second local process creates this alias after the existence check.
            # Both names and the protected file remain inside this temporary root.
            self.late_hardlink(target)
            return {"target_encoder": {"module.backbone.layer": 1}}
        self.torch.load.side_effect = load
        try:
            prepare.convert_checkpoint("fake-full.pt", target, "vjepa", self.claims, self.conversion)
        except (ValueError, FileExistsError):
            pass
        self.assertEqual(self.sentinel.read_bytes(), self.original,
                         "conversion must not truncate a late-created alias of an external file")

    def test_cli_unsafe_combinations_fail_before_download_or_source_preparation(self):
        combinations = [[], ["--sources-only"],
                        ["--accept-noncommercial", "--sources-only", "--download-checkpoints"],
                        ["--accept-noncommercial", "--sources-only", "--write-config", str(self.root / "new.toml")],
                        ["--accept-noncommercial", "--vjepa-full", "missing.pt"]]
        with patch.object(prepare, "prepare_source") as source, contextlib.redirect_stderr(io.StringIO()):
            for args in combinations:
                with self.subTest(args=args):
                    with self.assertRaises(SystemExit) as error:
                        prepare.main(args)
                    self.assertEqual(error.exception.code, 2)
            source.assert_not_called()
        self.urlopen.assert_not_called()
        self.torch.load.assert_not_called()

    def test_existing_config_is_not_overwritten_and_blocks_resource_work(self):
        config = self.outside / "runtime.toml"
        config.write_bytes(b"existing configuration")
        args = self.full_args(config)
        with patch.object(prepare, "prepare_source") as source, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                prepare.main(args)
            self.assertEqual(error.exception.code, 2)
            source.assert_not_called()
        self.assertEqual(config.read_bytes(), b"existing configuration")
        self.urlopen.assert_not_called()

    def test_native_config_is_valid_toml_with_exact_verified_resource_addresses(self):
        config = self.root / "new/runtime.toml"
        self.assertEqual(prepare.main(self.full_args(config)), 0)
        value = tomllib.loads(config.read_text(encoding="utf-8"))
        self.assertEqual(value["runtime"]["transport"], "native")
        self.assertEqual(value["runtime"]["python"], sys.executable)
        self.assertEqual(value["demo"]["default_algorithm"], "optimized")
        self.assertEqual(set(value["resources"]), {"vjepa_source", "ijepa_source", "vjepa_encoder",
                                                  "vjepa_predictor", "ijepa_checkpoint", "r3d_checkpoint"})
        for key, path in value["resources"].items():
            self.assertTrue(Path(path).is_relative_to(self.directory))
            if key.endswith("source"):
                self.assertEqual(source_tree_digest(path), self.claims[key]["source_tree_sha256"])
            else:
                self.assertEqual(prepare.digest_file(path), self.claims[key]["sha256"])
        self.urlopen.assert_not_called()
        self.torch.load.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "Windows-host WSL configuration boundary")
    def test_windows_wsl_config_does_not_embed_a_host_only_python_executable(self):
        config = self.root / "wsl-runtime.toml"
        args = self.full_args(config, transport="wsl")
        with patch.object(prepare.sys, "executable", r"C:\HostOnly\python.exe"):
            try:
                prepare.main(args)
            except SystemExit as error:
                self.assertEqual(error.code, 2)
                return  # Rejecting an unspecified WSL interpreter is also safe.
        value = tomllib.loads(config.read_text(encoding="utf-8"))
        self.assertFalse(PureWindowsPath(value["runtime"]["wsl_python"]).is_absolute(),
                         "the WSL interpreter must not be the Windows preparation interpreter")
        self.urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
