import unittest

import numpy as np

from jepa_consensus_denoising import (
    ConsensusDenoiseConfig,
    compute_consensus_score,
    denoise_labels_with_consensus,
    denoise_records_for_fold,
)
from train_segment_locator import VideoRecord


class JEPAConsensusDenoisingTests(unittest.TestCase):
    def test_consensus_score_uses_rank_channels_and_agreement(self):
        signals = np.array(
            [
                [0.2, 0.4, 0.6, 1.0],
                [0.8, 0.8, 0.8, 0.5],
            ],
            dtype=np.float32,
        )

        score = compute_consensus_score(
            signals,
            feature_names=["vjepa", "ijepa", "dual", "agree"],
            evidence_names=["vjepa", "ijepa", "dual"],
            agreement_name="agree",
            agreement_weight=0.5,
        )

        np.testing.assert_allclose(score, np.array([0.4, 0.6], dtype=np.float32), atol=1e-6)

    def test_denoise_removes_low_consensus_positive_island(self):
        labels = np.array([0, 1, 1, 0, 0], dtype=np.int64)
        consensus = np.array([0.1, 0.05, 0.1, 0.2, 0.1], dtype=np.float32)
        config = ConsensusDenoiseConfig(add_threshold=0.9, remove_threshold=0.2, min_add_length=2, min_keep_length=1)

        denoised, stats = denoise_labels_with_consensus(labels, consensus, config)

        np.testing.assert_array_equal(denoised, np.array([0, 0, 0, 0, 0], dtype=np.int64))
        self.assertEqual(stats["removed_frames"], 2)

    def test_denoise_adds_high_consensus_unlabeled_island(self):
        labels = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        consensus = np.array([0.1, 0.95, 0.9, 0.2, 0.1], dtype=np.float32)
        config = ConsensusDenoiseConfig(add_threshold=0.85, remove_threshold=0.1, min_add_length=2, min_keep_length=1)

        denoised, stats = denoise_labels_with_consensus(labels, consensus, config)

        np.testing.assert_array_equal(denoised, np.array([0, 1, 1, 0, 0], dtype=np.int64))
        self.assertEqual(stats["added_frames"], 2)

    def test_denoise_records_for_fold_preserves_validation_labels(self):
        records = [
            VideoRecord(
                name="train",
                signals=np.array([[0.1, 1.0], [0.9, 1.0], [0.9, 1.0]], dtype=np.float32),
                labels=np.array([0, 0, 0], dtype=np.int64),
            ),
            VideoRecord(
                name="val",
                signals=np.array([[0.9, 1.0], [0.9, 1.0], [0.1, 1.0]], dtype=np.float32),
                labels=np.array([1, 0, 0], dtype=np.int64),
            ),
        ]
        config = ConsensusDenoiseConfig(add_threshold=0.8, remove_threshold=0.1, min_add_length=2, min_keep_length=1)

        denoised, summary = denoise_records_for_fold(
            records,
            train_idx=[0],
            feature_names=["jepa", "agree"],
            evidence_names=["jepa"],
            agreement_name="agree",
            config=config,
        )

        np.testing.assert_array_equal(denoised[0].labels, np.array([0, 1, 1], dtype=np.int64))
        np.testing.assert_array_equal(denoised[1].labels, records[1].labels)
        self.assertEqual(summary["changed_videos"], 1)


if __name__ == "__main__":
    unittest.main()
