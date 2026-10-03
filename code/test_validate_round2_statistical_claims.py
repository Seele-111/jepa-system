import unittest

from validate_round2_statistical_claims import _assert_close


class StatisticalClaimValidationTests(unittest.TestCase):
    def test_assert_close_accepts_rounding_error(self):
        _assert_close(0.8183627, 0.81836, "metric")

    def test_assert_close_rejects_drift(self):
        with self.assertRaises(AssertionError):
            _assert_close(0.80, 0.82, "metric")


if __name__ == "__main__":
    unittest.main()
