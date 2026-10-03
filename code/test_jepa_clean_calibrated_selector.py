import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from jepa_clean_calibrated_selector import (
    combine_clean_and_raw_scores,
    fit_clean_candidate_calibrator,
    select_clean_calibrated_event_set_params,
)


class JEPACleanCalibratedSelectorTests(unittest.TestCase):
    def test_clean_score_can_override_raw_only_false_candidate(self):
        raw_scores = np.array([0.35, 0.9], dtype=np.float32)
        clean_scores = np.array([0.95, 0.05], dtype=np.float32)

        combined = combine_clean_and_raw_scores(
            raw_scores,
            clean_scores,
            clean_weight=0.8,
            raw_weight=0.2,
        )

        self.assertGreater(float(combined[0]), float(combined[1]))

    def test_selection_adds_clean_supported_rescue_without_fp_increase(self):
        base = [[(0, 4)]]
        candidates = [[(10, 14), (20, 22)]]
        raw_scores = [np.array([0.35, 0.9], dtype=np.float32)]
        clean_scores = [np.array([0.95, 0.05], dtype=np.float32)]
        labels = [
            np.array(
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=np.int64,
            )
        ]

        config, metrics, predictions = select_clean_calibrated_event_set_params(
            base,
            candidates,
            raw_scores,
            clean_scores,
            labels,
            clean_thresholds=[0.5],
            raw_thresholds=[0.0],
            clean_weights=[0.8],
            raw_weights=[0.2],
            max_base_ious=[0.0],
            length_penalties=[0.0],
            max_rescues_per_videos=[1],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertTrue(config["enabled"])
        self.assertEqual(predictions, [[(0, 4), (10, 14)]])
        self.assertEqual(metrics["segment"]["tp"], 2)
        self.assertEqual(metrics["segment"]["fp"], 0)

    def test_fp_guard_rejects_clean_supported_false_positive(self):
        base = [[(0, 4)]]
        candidates = [[(10, 14)]]
        raw_scores = [np.array([0.8], dtype=np.float32)]
        clean_scores = [np.array([0.9], dtype=np.float32)]
        labels = [
            np.array(
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                dtype=np.int64,
            )
        ]

        config, metrics, predictions = select_clean_calibrated_event_set_params(
            base,
            candidates,
            raw_scores,
            clean_scores,
            labels,
            clean_thresholds=[0.5],
            raw_thresholds=[0.0],
            clean_weights=[0.8],
            raw_weights=[0.2],
            max_base_ious=[0.0],
            length_penalties=[0.0],
            max_rescues_per_videos=[1],
            max_fp_increase=0,
            iou_threshold=0.3,
        )

        self.assertFalse(config["enabled"])
        self.assertEqual(predictions, base)
        self.assertEqual(metrics["segment"]["fp"], 0)
        self.assertGreater(config["diagnostics"]["fp_rejected_configs"], 0)

    def test_fit_clean_candidate_calibrator_excludes_validation_video_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_names = ["true_vjepa_raw_rank", "true_ijepa_dense_raw_rank", "dual_jepa_composite_rank"]
            np.savez_compressed(
                root / "signals.npz",
                np.array([[0.0, 0.0, 0.0], [0.9, 0.8, 0.7], [0.9, 0.8, 0.7]], dtype=np.float32),
                np.array([[0.0, 0.0, 0.0], [0.9, 0.8, 0.7], [0.9, 0.8, 0.7]], dtype=np.float32),
            )
            np.savez_compressed(
                root / "labels.npz",
                np.array([0, 1, 1], dtype=np.int64),
                np.array([0, 1, 1], dtype=np.int64),
            )
            (root / "video_names.json").write_text(json.dumps(["keep.mp4", "drop.mp4"]), encoding="utf-8")
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "n_videos": 2,
                        "frames": 6,
                        "positive_frames": 4,
                        "feature_dim": len(feature_names),
                        "feature_names": feature_names,
                    }
                ),
                encoding="utf-8",
            )

            bundle = fit_clean_candidate_calibrator(
                root,
                channel_names=["true_vjepa_raw_rank"],
                thresholds=[0.5],
                min_gaps=[0],
                min_lengths=[1],
                iou_threshold=0.3,
                model_name="extratrees",
                target="binary",
                seed=7,
                exclude_video_names={"drop.mp4"},
            )

        self.assertEqual(bundle.n_train_videos, 1)
        self.assertEqual(bundle.excluded_video_names, ["drop.mp4"])
        self.assertEqual(bundle.clean_video_names, ["keep.mp4"])


if __name__ == "__main__":
    unittest.main()
