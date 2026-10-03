"""Preserve native baseline video precision; no state/model fitting required."""
import unittest
import numpy as np
from select_baseline_semantic_precision import original_positive


class FakeModel:
    classes_=np.array([0,1])
    def predict_proba(self,x):
        p=np.full(len(x),0.123456789123456,np.float64)
        return np.stack([1-p,p],axis=1)


class PrecisionTests(unittest.TestCase):
    def test_native_video_not_float32_rounded(self):
        model=FakeModel();x=np.zeros((3,2));actual=original_positive(model,x)
        self.assertEqual(actual.dtype,np.float64)
        self.assertTrue(np.array_equal(actual,model.predict_proba(x)[:,1]))
        self.assertTrue(np.all(actual!=actual.astype(np.float32).astype(np.float64)))

    def test_no_positive_class_zero_original_precision(self):
        model=FakeModel();model.classes_=np.array([0])
        p=original_positive(model,np.zeros((2,2)))
        self.assertEqual(p.dtype,np.float64);self.assertTrue(np.array_equal(p,np.zeros(2)))


if __name__=='__main__':unittest.main()
