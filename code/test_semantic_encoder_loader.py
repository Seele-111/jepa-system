"""Storage repair safety: strict weights, CPU-only checkpoint and BF16 parity."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from optimized_semantic_encoder_loader import load_encoder_cpu_checkpoint


class LoaderTests(unittest.TestCase):
    def factory(self, *, pretrained):
        self.assertFalse(pretrained)
        return torch.nn.Linear(3, 2), torch.nn.Linear(2, 2)

    def test_same_weights_bf16_and_eval(self):
        torch.manual_seed(91)
        original = torch.nn.Linear(3, 2)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'trusted.pt'
            torch.save({'encoder': original.state_dict()}, path)
            with patch('torch.load', wraps=torch.load) as loader:
                actual = load_encoder_cpu_checkpoint(self.factory, path, 'cpu')
            self.assertEqual(loader.call_args.kwargs,
                             {'map_location':'cpu','weights_only':True,'mmap':True})
            self.assertFalse(actual.training)
            for name, weight in actual.state_dict().items():
                self.assertEqual(weight.dtype, torch.bfloat16)
                self.assertEqual(weight.device.type, 'cpu')
                self.assertTrue(torch.equal(weight, original.state_dict()[name].to(torch.bfloat16)))
            x = torch.tensor([[.5, -1., .75]], dtype=torch.bfloat16)
            self.assertTrue(torch.equal(actual(x), original.to(torch.bfloat16).eval()(x)))

    def test_strict_missing_and_extra_weights(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'trusted.pt'
            for state in ({'weight':torch.zeros(2, 3)},
                          {'weight':torch.zeros(2, 3),'bias':torch.zeros(2),'extra':torch.zeros(1)}):
                torch.save({'encoder':state}, path)
                with self.assertRaises(RuntimeError):
                    load_encoder_cpu_checkpoint(self.factory, path, 'cpu')

    def test_wrong_checkpoint_schema_is_not_silently_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'trusted.pt'
            torch.save({'ema_encoder':torch.nn.Linear(3, 2).state_dict()}, path)
            with self.assertRaises(ValueError):
                load_encoder_cpu_checkpoint(self.factory, path, 'cpu')


if __name__ == '__main__':
    unittest.main()
