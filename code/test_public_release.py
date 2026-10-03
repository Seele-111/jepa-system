"""Public distribution integrity, relocation and configuration regression tests."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from optimized_locator import load_bundle
from published_models import ROOT, digest_json, resolve_profiles, validate_profiles, validate_source, source_tree_digest
from jepa_runtime import settings, model_command


class PublicReleaseTests(unittest.TestCase):
    def test_both_reviewed_readouts_load_without_private_data(self):
        for name in ("locator", "motion"):
            with self.subTest(name=name):
                model = load_bundle(ROOT / "models" / "public" / (name + ".json"))
                self.assertEqual(model["publication"]["model_id"], name)
                self.assertNotIn("training_video_names", model)
                self.assertNotIn("training_video_sha256", model)
                self.assertNotIn("report_sources", model)
                self.assertEqual(model["publication"]["training_membership"], "not_distributed")

    def test_public_model_tampering_cannot_be_fixed_by_rehashing_metadata(self):
        model = load_bundle(ROOT / "models" / "public" / "motion.json")
        model["decoder"]["threshold"] += .01
        model["publication"]["inference_payload_sha256"] = digest_json({k: v for k, v in model.items() if k != "publication"})
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "edited.json"
            target.write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reviewed release"):
                load_bundle(target)

    def test_rgb_feature_changes_are_rejected_even_with_a_new_profile_id(self):
        model = load_bundle(ROOT / "models" / "public" / "locator.json")
        profiles = copy.deepcopy(model["feature_profiles"])
        profiles["rgb"]["anchor_interval_seconds"] = .4
        profiles["rgb"]["profile_id"] = digest_json({k: v for k, v in profiles["rgb"].items() if k != "profile_id"})
        with self.assertRaisesRegex(ValueError, "reviewed release"):
            validate_profiles(profiles)

    def test_only_resource_addresses_relocate_without_changing_the_input(self):
        model = load_bundle(ROOT / "models" / "public" / "locator.json")
        profiles = model["feature_profiles"]
        before = copy.deepcopy(profiles)
        result = resolve_profiles(profiles)
        self.assertEqual(profiles, before)
        self.assertEqual(result["motion"], profiles["motion"])
        for key in ("preprocess", "config", "feature_names", "extractor_sha256", "adapter_sha256"):
            if key in profiles["corrected"]:
                self.assertEqual(result["corrected"][key], profiles["corrected"][key])
        self.assertEqual(result["rgb"]["checkpoint_sha256"], profiles["rgb"]["checkpoint_sha256"])
        self.assertEqual(result["rgb"]["runtime_versions"], profiles["rgb"]["runtime_versions"])
        self.assertTrue(Path(result["rgb"]["checkpoint_path"]).is_absolute())
        self.assertTrue(Path(result["rgb"]["preprocess"]["baseline_path"]).is_absolute())

    def test_incomplete_source_cannot_be_consumed_despite_matching_python_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "src" / "tiny.py"
            source.parent.mkdir()
            source.write_text("VALUE = 1\n", encoding="utf-8")
            claim = {"resources": {"vjepa_source": {"source_tree_sha256": source_tree_digest(root)}}}
            with patch("published_models.registry", return_value=claim):
                with self.assertRaisesRegex(ValueError, "LICENSE"):
                    validate_source("vjepa", root)
                (root / "LICENSE").write_text("fixture license", encoding="utf-8")
                self.assertEqual(validate_source("vjepa", root), claim["resources"]["vjepa_source"])
                (root / ".jepa-preparation-incomplete").write_text("rejected", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    validate_source("vjepa", root)

    def test_release_has_no_original_user_paths_or_dataset_metadata(self):
        for name in ("locator.json", "motion.json", "registry.json"):
            text = (ROOT / "models" / "public" / name).read_text(encoding="utf-8")
            for marker in ("/home/zzy", "Users/admin", "Desktop", "training_video_names", "training_video_sha256", "report_sources"):
                self.assertNotIn(marker, text)


class RuntimeConfigTests(unittest.TestCase):
    def make_config(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "runtime.toml"
        path.write_text(text, encoding="utf-8")
        self.enterContext(patch.dict(os.environ, {"JEPA_CONFIG": str(path)}))
        return path

    def test_unknown_field_does_not_silently_select_defaults(self):
        self.make_config('[runtime]\nworker_potr=5006\n')
        with self.assertRaisesRegex(ValueError, "unknown field"):
            settings()

    def test_explicit_native_python_and_model_commands_use_no_shell(self):
        self.make_config('[runtime]\ntransport="native"\npython="custom python"\nworker_port=5006\nretain_models=false\n')
        cfg = settings()
        self.assertEqual(cfg.worker_port, 5006)
        self.assertFalse(cfg.retain_models)
        command = model_command(ROOT / "code" / "optimized_model_worker.py", ["--video", "a b.mp4"])
        self.assertEqual(command[0], "custom python")
        self.assertIn("--config", command)
        self.assertEqual(command[-1], "a b.mp4")

    def test_explicit_wsl_runtime(self):
        self.make_config('[runtime]\ntransport="wsl"\nwsl_distro="ExampleDistro"\nwsl_python="/opt/env/bin/python"\n')
        command = model_command(ROOT / "code" / "optimized_model_worker.py", [])
        self.assertEqual(command[:5], ["wsl.exe", "-d", "ExampleDistro", "--", "/opt/env/bin/python"])

    def test_loopback_port_range_and_boolean_retention_are_validated(self):
        for text in ('[runtime]\nworker_port=80\n', '[runtime]\nretain_models="false"\n'):
            with self.subTest(text=text):
                self.make_config(text)
                with self.assertRaises(ValueError):
                    settings()


if __name__ == "__main__":
    unittest.main()
