import unittest

from cross_validate_mil_event_locator import make_fold_args


class CrossValidateMILEventLocatorTests(unittest.TestCase):
    def test_make_fold_args_offsets_seed_without_mutating_base_args(self):
        class Args:
            seed = 42
            other = "value"

        fold_args = make_fold_args(Args(), fold_idx=3)

        self.assertEqual(fold_args.seed, 45)
        self.assertEqual(fold_args.other, "value")
        self.assertEqual(Args.seed, 42)


if __name__ == "__main__":
    unittest.main()
