"""v7 score fusion and nontrivial budget selection behavior checks."""
import unittest
from unittest.mock import patch
import numpy as np
import run_multiview_experiment_v7 as V7
from run_optimization_selection import Probabilities


class MultiViewTests(unittest.TestCase):
    def records(self):
        normal={'sha256':'normal','labels':np.zeros(30,np.uint8),'fps':20.,'frames':30,'event_count':0}
        labels=np.zeros(30,np.uint8);labels[10:20]=1
        abnormal={'sha256':'abnormal','labels':labels,'fps':20.,'frames':30,'event_count':1}
        return [normal,abnormal]

    def test_linear_fusion_is_fixed_and_aligned(self):
        a=Probabilities({0:np.array([.2,.6],np.float32)},{0:.4},None)
        b=Probabilities({0:np.array([.8,.4],np.float32)},{0:.8},None)
        p=V7.fuse(a,b,[0])
        np.testing.assert_array_equal(p.frame[0],np.array([.5,.5],np.float32))
        self.assertAlmostEqual(p.video[0],.6)
        np.testing.assert_array_equal(a.frame[0],np.array([.2,.6],np.float32))

    def test_invalid_fusion_fails_closed(self):
        p=Probabilities({0:np.array([.2],np.float32)},{0:.4},None)
        for q in [Probabilities({0:np.array([.2,.3],np.float32)},{0:.4},None),
                  Probabilities({0:np.array([np.nan],np.float32)},{0:.4},None),
                  Probabilities({0:np.array([.3],np.float32)},{},None)]:
            with self.assertRaises(ValueError):V7.fuse(p,q,[0])

    def test_budget_inclusive_boundary_and_normalized_excess(self):
        stats=np.zeros(12);stats[9]=3;stats[10]=2
        result=V7.error_budget(stats,10,10);self.assertTrue(result['feasible'])
        stats[9]=6;stats[10]=4;result=V7.error_budget(stats,10,10)
        self.assertFalse(result['feasible']);self.assertAlmostEqual(result['normalized_violation'],2.)

    def test_feasible_decoder_wins_despite_f1_penalty(self):
        records=self.records();configs=[{'id':0},{'id':1},{'id':2}]
        preds={0:{0:[(10,19)],1:[(10,19)]},1:{0:[],1:[]},2:{0:[],1:[(11,18)]}}
        with patch.object(V7,'configurations',return_value=configs),patch.object(V7,'decode_predictions',side_effect=lambda r,ids,p,c:preds[c['id']]):
            selected=V7.choose(records,[0,1],None,101)
        self.assertEqual(selected['config']['id'],2);self.assertTrue(selected['error_budget']['feasible'])

    def test_infeasible_fallback_reports_violation(self):
        records=self.records();configs=[{'id':0},{'id':1}]
        preds={0:{0:[(10,19)],1:[(10,19)]},1:{0:[],1:[]}}
        with patch.object(V7,'configurations',return_value=configs),patch.object(V7,'decode_predictions',side_effect=lambda r,ids,p,c:preds[c['id']]):
            selected=V7.choose(records,[0,1],None,101)
        self.assertEqual(selected['config']['id'],0);self.assertFalse(selected['error_budget']['feasible'])
        self.assertAlmostEqual(selected['error_budget']['normalized_violation'],(1-.3)/.3)

    def test_duplicate_predictions_keep_first_config(self):
        records=self.records();pred={0:[],1:[(10,19)]}
        with patch.object(V7,'configurations',return_value=[{'id':0},{'id':1}]),patch.object(V7,'decode_predictions',return_value=pred):
            selected=V7.choose(records,[0,1],None,101)
        self.assertEqual(selected['config']['id'],0);self.assertEqual(len(selected['shortlist']),1)

if __name__=='__main__':unittest.main()
