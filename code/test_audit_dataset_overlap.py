import unittest

from audit_dataset_overlap import audit


class DatasetOverlapTests(unittest.TestCase):
    def test_audit_reports_pairwise_overlap(self):
        report = audit({"a": {"x", "y"}, "b": {"y", "z"}, "c": {"q"}})
        row = next(item for item in report["pairs"] if item["left"] == "a" and item["right"] == "b")
        self.assertEqual(row["overlap"], ["y"])
        self.assertFalse(report["zero_overlap"])


if __name__ == "__main__": unittest.main()
