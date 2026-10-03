import unittest
import numpy as np
from optimized_locator import decode,metrics
from optimized_recall_decoder import decode_v2,effective_score
from optimized_probability_calibration import fit_logit_calibration,apply_calibration,fit_oof_calibration
from optimized_feature_view_v2 import feature_view_v2
from optimized_video_statistics import VideoTruth,video_statistics,utility,stratified_bootstrap_counts

class RecallV2Tests(unittest.TestCase):
    def test_seed_requires_sustained_evidence_not_one_peak_on_platform(self):
        p=np.array([.5,.5,.9,.5,.5],np.float32);c={'threshold':.8,'low_ratio':.6,'seed_seconds':.2}
        self.assertEqual(decode_v2(p,10,1,c),[])
        c['seed_seconds']=0;self.assertEqual(decode_v2(p,10,1,c),[(0,4)])
    def test_weak_video_floor_does_not_delete_local_evidence(self):
        p=np.full(8,.9,np.float32);c={'threshold':.5,'video_strength':.5}
        self.assertEqual(decode_v2(p,10,0,c),[])
        c['video_floor']=.5;self.assertEqual(decode_v2(p,10,0,c),[(0,7)])
    def test_minimum_real_seconds_uses_ceil(self):
        self.assertEqual(decode_v2([.9,.9],10,1,{'threshold':.5,'min_seconds':.25}),[])
    def test_v1_parity_without_new_options_and_duration_floor(self):
        rng=np.random.default_rng(8)
        for _ in range(50):
            p=rng.random(60).astype(np.float32);v=float(rng.random());c={'threshold':.5,'low_ratio':.7,'smooth_seconds':.15,'gap_seconds':.12,'min_seconds':0,'video_strength':.5,'video_threshold':.35}
            self.assertEqual(decode_v2(p,16,v,c),decode(p,16,v,c))
    def test_invalid_modulation_and_nan_fail_closed(self):
        for c in [{'threshold':.5,'video_floor':-1},{'threshold':.5,'seed_seconds':-1},{'threshold':float('nan')}]:
            with self.assertRaises(ValueError):decode_v2([.9],10,1,c)
        with self.assertRaises(ValueError):decode_v2([np.nan],10,1,{'threshold':.5})

class CalibrationTests(unittest.TestCase):
    def test_weighted_empirical_prior_can_raise_video_score(self):
        state=fit_logit_calibration(np.full(20,.5),np.r_[np.ones(16),np.zeros(4)])
        self.assertGreater(float(apply_calibration([.5],state)[0]),.65)
    def test_monotone_finite_at_endpoints(self):
        state=fit_logit_calibration([0,.1,.2,.5,.7,1],[0,0,0,1,1,1])
        p=apply_calibration(np.linspace(0,1,100),state)
        self.assertTrue(np.isfinite(p).all());self.assertTrue(np.all(np.diff(p)>=0))
    def test_single_class_identity_and_oof_partition_guard(self):
        self.assertEqual(fit_logit_calibration([.3,.6],[1,1])['kind'],'identity')
        with self.assertRaises(ValueError):fit_oof_calibration([{'labels':np.array([1])}],[0],{0:np.array([.5])},{1:.5})
    def test_invalid_calibration_inputs_rejected(self):
        for p,y,w in [([np.nan],[1],None),([.5],[2],None),([.5],[1],[0])]:
            with self.assertRaises(ValueError):fit_logit_calibration(p,y,w)

class V2FeatureTests(unittest.TestCase):
    def test_no_global_statistics_or_flag_derivative_in_frame_view(self):
        r={'fps':10,'motion':np.column_stack([np.arange(8),np.r_[0,np.ones(7)]]).astype(np.float32),'motion_names':['speed','motion_valid']}
        x,v,n=feature_view_v2('noglobal_motion_et',r)
        self.assertEqual(len(v),8);self.assertEqual(x.shape[1],14)
        self.assertNotIn('speed/global_mean',n);self.assertEqual([a for a in n if a.startswith('motion_valid')],['motion_valid'])
        r['labels']=np.zeros(8);r['name']='anything';x2,_,_=feature_view_v2('noglobal_motion_et',r)
        np.testing.assert_array_equal(x,x2)
    def test_flag_and_alignment_schema_guards(self):
        r={'fps':10,'motion':np.zeros((8,1)),'motion_names':['speed'],'local':np.zeros((7,1)),'local_names':['local_speed']}
        with self.assertRaises(ValueError):feature_view_v2('local_motion_et',r)
        r={'fps':10,'motion':np.full((8,1),2),'motion_names':['motion_valid']}
        with self.assertRaises(ValueError):feature_view_v2('noglobal_motion_et',r)

class StatisticsTests(unittest.TestCase):
    def test_inclusive_greedy_and_frame_union_matches_reference(self):
        rng=np.random.default_rng(19)
        for _ in range(200):
            y=rng.random(50)>.6
            pred=[(int(a),int(b)) for a,b in zip(rng.integers(-3,35,4),rng.integers(35,55,4))]
            stats=video_statistics(pred,VideoTruth(y));reference=metrics([pred],[y])
            a=reference['iou_0.3'];b=reference['iou_0.5'];f=reference['frame']
            self.assertEqual(stats.tolist()[:9],[a['tp'],a['fp'],a['fn'],b['tp'],b['fp'],b['fn'],f['tp'],f['fp'],f['fn']])
    def test_bootstrap_preserves_video_strata_and_is_deterministic(self):
        records=[{'event_count':v} for v in [0,0,1,1,2,3]];indices=list(range(6))
        a=stratified_bootstrap_counts(records,indices,99);b=stratified_bootstrap_counts(records,indices,99)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[:,:2].sum(1),2)
        self.assertTrue(np.all(a.sum(1)==6))

if __name__=='__main__':unittest.main()
