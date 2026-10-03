import csv
import json
import tempfile
import unittest
from pathlib import Path

from build_review_suggestion_draft import build_suggestion_draft, suggest_row


class BuildReviewSuggestionDraftTests(unittest.TestCase):
    def test_suggest_row_uses_high_iou_candidate_for_split_parent(self):
        row = {
            "action": "review_split_or_relabel_parent",
            "target_segment": [10, 20],
            "current_prediction": [0, 80],
            "best_jepa_candidate": [11, 19],
            "best_candidate_iou": 0.8,
        }

        suggestion = suggest_row(row)

        self.assertEqual(suggestion["suggested_decision"], "replace_video_segments")
        self.assertEqual(suggestion["suggested_error_segments"], [[11, 19]])
        self.assertIn("人工确认", suggestion["suggestion_note"])

    def test_suggest_row_keeps_low_confidence_as_uncertain(self):
        row = {
            "action": "review_parent_boundary_and_matching",
            "target_segment": [10, 20],
            "best_jepa_candidate": [0, 50],
            "best_candidate_iou": 0.2,
        }

        suggestion = suggest_row(row)

        self.assertEqual(suggestion["suggested_decision"], "uncertain")

    def test_build_suggestion_draft_writes_csv_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review = {
                "items": [
                    {
                        "kind": "FN",
                        "priority": 200,
                        "video": "a.mp4",
                        "video_idx": 0,
                        "action": "review_split_or_relabel_parent",
                        "target_segment": [10, 20],
                        "current_prediction": [0, 80],
                        "best_jepa_candidate": [11, 19],
                        "best_candidate_iou": 0.8,
                    }
                ]
            }
            review_json = root / "review.json"
            review_json.write_text(json.dumps(review), encoding="utf-8")

            result = build_suggestion_draft(review_json, root / "draft", top_k=1)

            self.assertEqual(result["n_items"], 1)
            self.assertTrue((root / "draft" / "review_suggestion_draft.csv").exists())
            self.assertTrue((root / "draft" / "review_suggestion_draft.json").exists())
            with (root / "draft" / "review_suggestion_draft.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["suggested_decision"], "replace_video_segments")


if __name__ == "__main__":
    unittest.main()
