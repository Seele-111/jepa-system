"""Regression for seeded I-JEPA hidden counter without loading checkpoints."""
import unittest
from contextlib import contextmanager
from optimized_jepa_extractor import seeded_mask_counter

class Counter:
    def __init__(self,value):self.value=value
    @contextmanager
    def get_lock(self):yield

class Collator:
    def __init__(self,value):self._itr_counter=Counter(value)
    def step(self):self._itr_counter.value+=1;return self._itr_counter.value

class CounterTests(unittest.TestCase):
    def test_prior_requests_do_not_change_explicit_seed(self):
        collator=Collator(341)
        with seeded_mask_counter(collator,2):self.assertEqual(collator.step(),2)
        self.assertEqual(collator._itr_counter.value,341)
        collator._itr_counter.value=711
        with seeded_mask_counter(collator,2):self.assertEqual(collator.step(),2)
        self.assertEqual(collator._itr_counter.value,711)
    def test_failure_restores_counter(self):
        collator=Collator(-1)
        with self.assertRaises(RuntimeError):
            with seeded_mask_counter(collator,0):
                self.assertEqual(collator.step(),0)
                raise RuntimeError('mask failure')
        self.assertEqual(collator._itr_counter.value,-1)
    def test_int32_seed_is_bounded(self):
        collator=Collator(0)
        with seeded_mask_counter(collator,2**62):
            self.assertEqual(collator.step(),2**62%(2**31-1))
    def test_unknown_collator_is_rejected(self):
        with self.assertRaises(RuntimeError):
            with seeded_mask_counter(object(),0):pass

if __name__=='__main__':unittest.main()
