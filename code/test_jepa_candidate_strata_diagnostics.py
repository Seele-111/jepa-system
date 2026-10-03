import unittest

import numpy as np

from jepa_candidate_strata_diagnostics import (
    classify_candidate_strata,
    summarize_strata_scores,
)


class JEPACandidateStrataDiagnosticsTests(unittest.TestCase):
    def test_classifies_oracle_rescue_false_positive_and_covered_positive(self):
        labels = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.int64)
        base = [(1, 2)]
        candidates = [(1, 2), (5, 6), (3, 4), (0, 7)]

        rows = classify_candidate_strata(
            base,
            candidates,
            labels,
            iou_threshold=0.3,
            base_overlap_iou=0.5,
        )

        self.assertEqual([row["stratum"] for row in rows], [
            "base_prediction",
            "oracle_rescue",
            "false_positive",
            "false_positive",
        ])
        self.assertAlmostEqual(rows[1]["best_iou"], 1.0)

    def test_summarize_strata_scores_reports_score_gap(self):
        rows = [
            {"stratum": "oracle_rescue", "selector_score": 0.9, "ot_score": 1.2},
            {"stratum": "oracle_rescue", "selector_score": 0.7, "ot_score": 1.0},
            {"stratum": "false_positive", "selector_score": 0.8, "ot_score": 0.2},
        ]

        summary = summarize_strata_scores(rows, score_keys=["selector_score", "ot_score"])

        self.assertEqual(summary["oracle_rescue"]["count"], 2)
        self.assertEqual(summary["false_positive"]["count"], 1)
        self.assertGreater(summary["oracle_rescue"]["ot_score"]["mean"], summary["false_positive"]["ot_score"]["mean"])


if __name__ == "__main__":
    unittest.main()
