"""Targeted cost change is equivalent to raw mass4 and preserves event balance."""
import unittest
import numpy as np
from optimized_normal_cost_v9 import normal_cost_weights
from optimized_event_training_v5 import event_weights


class NormalCostTests(unittest.TestCase):
    def records(self):
        y=np.zeros(40,np.uint8);y[5:7]=1;y[15:25]=1
        return [{'sha256':'normal','labels':np.zeros(20,np.uint8)},
                {'sha256':'event','labels':y}]

    def test_normal_vs_abnormal_background_mass_is_four(self):
        r=self.records();w=normal_cost_weights(r,[0,1]);back=w[20:][r[1]['labels']==0].sum()
        self.assertAlmostEqual(w[:20].sum()/back,4.)
        self.assertAlmostEqual(w[25:27].sum(),w[35:45].sum())
        labels=np.concatenate([x['labels'] for x in r])
        self.assertAlmostEqual(w[labels>0].sum(),w[labels==0].sum());self.assertAlmostEqual(w.mean(),1.)

    def test_duplicate_normal_content_splits_same_mass(self):
        r=self.records();a=normal_cost_weights(r,[0,1]);r.append({'sha256':'normal','labels':r[0]['labels'].copy()})
        b=normal_cost_weights(r,[0,1,2]);self.assertAlmostEqual((b[:20].sum()+b[60:].sum())/b[20:60][r[1]['labels']==0].sum(),4.)
        np.testing.assert_array_equal(b[:20],b[60:])

    def test_class_degenerate_input_fails_closed(self):
        with self.assertRaises(ValueError):normal_cost_weights([{'sha256':'n','labels':np.zeros(20,np.uint8)}],[0])

if __name__=='__main__':unittest.main()
