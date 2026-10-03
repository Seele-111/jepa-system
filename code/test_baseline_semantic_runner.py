"""Runner isolation, fixed mode selection and fail-closed probability checks."""
import unittest
from unittest.mock import patch
import numpy as np
import run_baseline_semantic_readout as R


class RunnerTests(unittest.TestCase):
    def test_index_and_alias_content_isolation(self):
        records=[{'sha256':'same'},{'sha256':'same'},{'sha256':'other'}]
        for fit,predict,excluded in (([0],[0],[]),([0],[1],[]),([0],[2],[1])):
            with self.assertRaises(ValueError):R.isolation(records,fit,predict,excluded)
        R.isolation(records,[0,1],[2],[])

    def test_complete_probabilities_and_nonfinite_rejection(self):
        records=[{'frames':3}];fp={0:np.array([.1,.2,.3],np.float32)};vp={0:.4}
        R.valid_probabilities(fp,vp,[0],records)
        for bad in (np.array([.1,.2]),np.array([.1,np.nan,.3],np.float32),np.array([.1,1.2,.3],np.float32),np.array([.1,.2,.3],np.float64)):
            with self.assertRaises(ValueError):R.valid_probabilities({0:bad},vp,[0],records)
        with self.assertRaises(ValueError):R.valid_probabilities(fp,{},[0],records)

    def test_selector_keeps_default_on_exact_tie(self):
        records=[{'sha256':'normal','event_count':0,'labels':np.zeros(5,bool)},
                 {'sha256':'event','event_count':1,'labels':np.array([0,1,1,0,0],bool)}]
        same={0:[],1:[(1,2)]}
        choice=R.choose_mode(records,[0,1],{mode:same for mode in R.MODES},101)
        self.assertEqual(choice['mode'],'default_control')
        self.assertEqual(len(choice['configurations']),2)

    def test_selector_uses_only_given_training_records(self):
        class HeldOut(dict):
            def __getitem__(self,key):raise AssertionError('heldout label read')
        records=[{'sha256':'normal','event_count':0,'labels':np.zeros(5,bool)},
                 {'sha256':'event','event_count':1,'labels':np.array([0,1,1,0,0],bool)},HeldOut()]
        predictions={'default_control':{0:[],1:[]},'semantic_content':{0:[],1:[(1,2)]}}
        choice=R.choose_mode(records,[0,1],predictions,101)
        self.assertEqual(choice['mode'],'semantic_content')
        with self.assertRaises(ValueError):R.choose_mode(records,[0,1],{mode:{0:[]} for mode in R.MODES},101)

    def test_promotions_require_all_existing_guards(self):
        protocol={'guards':{'event_f1_03_min':.5034013605442177,'event_f1_05_min':.34653061224489793,
            'frame_f1_min':.5586221701795472,'normal_fp_max':5,'positive_empty_max':12}}
        exact={'iou_0.3':{'f1':.5034013605442177},'iou_0.5':{'f1':.34653061224489793},
            'frame':{'f1':.5586221701795472},'normal':{'false_positive_videos':5},'positive_videos_without_candidate':12}
        self.assertTrue(all(R.guards(exact,protocol).values()))
        for path,bad in ((('iou_0.3','f1'),.50),(('iou_0.5','f1'),.34),(('frame','f1'),.55),
                         (('normal','false_positive_videos'),6),(('positive_videos_without_candidate',),13)):
            import copy
            changed=copy.deepcopy(exact)
            if len(path)==1:changed[path[0]]=bad
            else:changed[path[0]][path[1]]=bad
            self.assertFalse(all(R.guards(changed,protocol).values()))


if __name__=='__main__':unittest.main()
