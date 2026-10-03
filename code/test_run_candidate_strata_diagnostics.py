import unittest

import numpy as np

from run_candidate_strata_diagnostics import build_strata_rows_for_video, summarize_all_strata


class RunCandidateStrataDiagnosticsTests(unittest.TestCase):
    def test_build_rows_attaches_selector_graph_and_ot_scores(self):
        labels = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)
        rows = build_strata_rows_for_video(
            fold=0,
            video_idx=7,
            video_name="toy.mp4",
            base_predictions=[(1, 2)],
            candidates=[(1, 2), (5, 6), (3, 4)],
            raw_scores=np.array([0.4, 0.7, 0.9], dtype=np.float32),
            active_scores=np.array([0.5, 0.8, 0.2], dtype=np.float32),
            ot_scores=np.array([1.0, 1.4, 0.1], dtype=np.float32),
            labels=labels,
            iou_threshold=0.3,
        )

        rescue = [row for row in rows if row["stratum"] == "oracle_rescue"]
        false_positive = [row for row in rows if row["stratum"] == "false_positive"]

        self.assertEqual(len(rows), 3)
        self.assertEqual(len(rescue), 1)
        self.assertEqual(rescue[0]["candidate"], [5, 6])
        self.assertAlmostEqual(rescue[0]["raw_score"], 0.7, places=5)
        self.assertAlmostEqual(rescue[0]["active_score"], 0.8, places=5)
        self.assertAlmostEqual(rescue[0]["ot_score"], 1.4, places=5)
        self.assertEqual(rescue[0]["raw_rank"], 2)
        self.assertEqual(rescue[0]["active_rank"], 1)
        self.assertEqual(rescue[0]["ot_rank"], 1)
        self.assertEqual(false_positive[0]["candidate"], [3, 4])

    def test_summarize_all_strata_reports_counts_and_score_gap(self):
        rows = [
            {"stratum": "oracle_rescue", "raw_score": 0.7, "active_score": 0.8, "ot_score": 1.4},
            {"stratum": "oracle_rescue", "raw_score": 0.6, "active_score": 0.7, "ot_score": 1.2},
            {"stratum": "false_positive", "raw_score": 0.9, "active_score": 0.2, "ot_score": 0.1},
        ]

        summary = summarize_all_strata(rows)

        self.assertEqual(summary["oracle_rescue"]["count"], 2)
        self.assertEqual(summary["false_positive"]["count"], 1)
        self.assertGreater(
            summary["oracle_rescue"]["ot_score"]["mean"],
            summary["false_positive"]["ot_score"]["mean"],
        )


if __name__ == "__main__":
    unittest.main()
