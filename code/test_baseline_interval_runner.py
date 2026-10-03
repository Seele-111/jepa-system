"""Risk-focused tests for strict cross-fit roles, sampling and inner-only choice."""
import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
import run_baseline_interval_quality as R
import optimized_baseline_crossfit as B


class SamplingContracts(unittest.TestCase):
    def test_sampling_preserves_all_three_strata_and_caps(self):
        y=np.r_[np.full(120,.02),np.full(110,.3),np.full(100,.8)].astype(np.float32)
        a=R.sample_indices(y,False,731);b=R.sample_indices(y,False,731)
        self.assertTrue(np.array_equal(a,b));self.assertEqual(len(a),96)
        self.assertEqual(sum(y[a]<.1),32);self.assertEqual(sum((y[a]>=.1)&(y[a]<.5)),32);self.assertEqual(sum(y[a]>=.5),32)
        self.assertEqual(len(set(a)),len(a));self.assertTrue(np.all(np.diff(a)>0))

    def test_normal_and_empty_sampling(self):
        self.assertEqual(len(R.sample_indices(np.zeros(500),True,5)),96)
        self.assertEqual(R.sample_indices(np.zeros(0),True,5).dtype,np.dtype(np.int32))
        self.assertEqual(len(R.sample_indices(np.zeros(0),False,5)),0)
        self.assertEqual(len(R.sample_indices(np.array([.1,.5]),False,5)),2)


class HeadPartitionContracts(unittest.TestCase):
    def setUp(self):
        self.records=[{'sha256':str(i),'labels':np.array([0,1],np.uint8),'fps':24.,'frames':2} for i in range(6)]

    def test_inner_uses_only_deep_scores_from_its_fit(self):
        f={i:np.array([.2,.6],np.float32) for i in range(4)};v={i:.5 for i in f}
        pred=R.Probabilities({i:np.array([.4,.7],np.float32) for i in range(6)},dict.fromkeys(range(6),.5),None)
        with patch.object(R,'parts',return_value=[{'fit':list(range(4)),'validation':[4,5]}]),\
             patch.object(R,'combine_jobs',return_value=(R.Probabilities(f,v,None),['deep'])) as combine,\
             patch.object(R,'legacy',return_value=(pred,{},['legacy'])):
            train,out,ids,val,fs,ps=R.head_inputs(0,0,self.records)
        self.assertEqual(set(train.frame),{0,1,2,3});self.assertEqual(set(out.frame),{4,5})
        self.assertEqual(combine.call_args.args[0],['s0_i0_d0','s0_i0_d1','s0_i0_d2'])

    def test_missing_deep_coverage_rejected(self):
        with patch.object(R,'parts',return_value=[{'fit':[0,1],'validation':[2,3]}]),\
             patch.object(R,'combine_jobs',return_value=(R.Probabilities({0:np.array([.2,.6])},{0:.5},None),[])),\
             patch.object(R,'legacy',return_value=(R.Probabilities({2:np.array([.2,.6]),3:np.array([.2,.6])},{2:.5,3:.5},None),{},[])):
            with self.assertRaisesRegex(ValueError,'coverage'):R.head_inputs(0,0,self.records)

    def test_alias_in_fit_and_prediction_rejected(self):
        records=copy.deepcopy(self.records);records[2]['sha256']=records[0]['sha256']
        with patch.object(R,'parts',return_value=[{'fit':[0,1],'validation':[2,3]}]),\
             patch.object(R,'combine_jobs',return_value=(R.Probabilities({0:np.array([.2,.6]),1:np.array([.2,.6])},{0:.5,1:.5},None),[])),\
             patch.object(R,'legacy',return_value=(R.Probabilities({2:np.array([.2,.6]),3:np.array([.2,.6])},{2:.5,3:.5},None),{},[])):
            with self.assertRaisesRegex(ValueError,'content leaks'):R.head_inputs(0,0,records)

    def test_combine_rejects_duplicate_deep_coverage(self):
        fake=({0:np.array([.2,.6])},{0:.5},{})
        with patch.object(B,'load_job',return_value=fake):
            with self.assertRaisesRegex(ValueError,'duplicate'):R.combine_jobs(['a','b'],self.records)


class InnerSelectionContracts(unittest.TestCase):
    def test_choice_does_not_read_outer_labels(self):
        records=[{'labels':np.array([1,1],np.uint8),'sha256':'a','event_count':1,'fps':24.,'frames':2},
                 {'labels':np.array([0,0],np.uint8),'sha256':'b','event_count':0,'fps':24.,'frames':2},
                 {'labels':None,'sha256':'excluded','event_count':None,'fps':None,'frames':None}]
        items={0:{'intervals':np.array([[0,1]],np.int32),'scores':np.array([.6],np.float32),'frame':np.array([.7,.7],np.float32),'video':.7},
               1:{'intervals':np.array([[0,1]],np.int32),'scores':np.array([.2],np.float32),'frame':np.array([.3,.3],np.float32),'video':.4}}
        protocol={'configurations':[{'enabled':False}]+[{'enabled':True,'threshold':t} for t in R.THRESHOLDS],
                  'default_decoder':{'threshold':.6,'low_ratio':.7,'smooth_seconds':0,'gap_seconds':0,'min_seconds':0,'video_threshold':0}}
        with patch.object(R,'check_protocol',return_value=protocol):
            first=R.choose(records,[0,1],items,177)
            records[2]={'labels':np.array([99]),'sha256':'adversarial','event_count':999,'fps':0,'frames':1}
            second=R.choose(records,[0,1],items,177)
        self.assertEqual(first,second);self.assertIn(first['config'],protocol['configurations'])

    def test_unchanged_original_guards_reject_bad_normal_or_empty(self):
        limits={'event_f1_03_min':.5034013605442177,'event_f1_05_min':.34653061224489793,
                'frame_f1_min':.5586221701795472,'normal_fp_max':5,'positive_empty_max':12}
        m={'iou_0.3':{'f1':.6},'iou_0.5':{'f1':.5},'frame':{'f1':.6},
           'normal':{'false_positive_videos':6},'positive_videos_without_candidate':13}
        with patch.object(R,'check_protocol',return_value={'guards':limits}):g=R.guards(m)
        self.assertFalse(g['normal_fp_no_worse']);self.assertFalse(g['positive_empty_improved_3'])
        self.assertTrue(g['event_f1_05_improved_02'])


if __name__=='__main__':unittest.main()