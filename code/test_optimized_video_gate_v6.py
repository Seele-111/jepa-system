import copy,importlib.util,unittest
import numpy as np
from optimized_video_gate_v6 import gate_features,fit_gate,apply_gate,GATE_KINDS
from test_optimized_compact_v4 import record
from run_video_gate_experiment_v6 import configurations,protocol,utility

class VideoGateFeatures(unittest.TestCase):
    def test_feature_count_and_no_label_or_name_leak(self):
        r=record();p=np.linspace(.1,.8,r['frames'],dtype=np.float32)
        x,n=gate_features(p,.7,r);self.assertEqual(len(n),41);self.assertEqual(len(set(n)),41)
        q=copy.deepcopy(r);q['labels'][:]=0;q.update(name='other',generator='other',sha256='different')
        z,m=gate_features(p,.7,q);np.testing.assert_array_equal(x,z);self.assertEqual(n,m)

    def test_bad_probability_fps_schema(self):
        r=record();p=np.full(r['frames'],.4,np.float32)
        for bad in [np.full(len(p),np.nan),np.full(len(p),1.1),p[:-1],p[:,None]]:
            with self.assertRaises(ValueError):gate_features(bad,.7,r)
        q=copy.deepcopy(r);q['fps']=0
        with self.assertRaises(ValueError):gate_features(p,.7,q)
        with self.assertRaises(ValueError):apply_gate({'kind':'other'},p,.7,r)

    def test_config_grid_and_risk_penalties(self):
        self.assertEqual(len(list(configurations())),60)
        self.assertTrue(all(c.get('video_strength',0)==0 for c in configurations()))
        s=np.zeros(12);s[9]=.5;s[10]=.5
        self.assertAlmostEqual(float(utility(s,.5,.5)),-.6)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'existing sklearn runtime required')
    def test_gate_prediction_and_training_partition_invariance(self):
        r=[record(i) for i in range(12)]
        for i in [0,3,6]:r[i]['labels'][:]=0
        f={i:np.linspace(.1,.4+(.3 if np.any(x['labels']) else 0),x['frames'],dtype=np.float32) for i,x in enumerate(r)}
        v={i:.5+(.3 if np.any(x['labels']) else -.1) for i,x in enumerate(r)}
        for kind in GATE_KINDS:
            state=fit_gate(kind,r,list(range(11)),f,v,12)
            q=apply_gate(state,f[11],v[11],r[11]);self.assertTrue(0<=q<=1)
            rr=copy.deepcopy(r);rr[11]['labels'][:]=0;rr[11]['corrected']*=10000
            other=fit_gate(kind,rr,list(range(11)),f,v,12)
            self.assertEqual(state,other)
            self.assertEqual(state['fit_content_sha256'],sorted(x['sha256'] for x in r[:11]))

if __name__=='__main__':unittest.main()
