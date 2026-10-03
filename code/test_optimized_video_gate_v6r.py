"""Independent sklearn reference and low-variance numeric regression tests."""
import importlib.util, unittest
from unittest.mock import patch
import numpy as np
from optimized_video_gate_v6r import apply_gate, repaired_state, fit_gate
from optimized_video_gate_v6 import apply_gate as old_apply, gate_features, GATE_KINDS
from optimized_detector import predict_record
from test_optimized_compact_v4 import record


def sklearn_reference(state, x):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.classes_ = np.asarray([0, 1]); model.coef_ = np.asarray([state['coefficients']], np.float64)
    model.intercept_ = np.asarray([state['intercept']], np.float64)
    model.n_features_in_ = len(x)
    z = np.clip((np.asarray(x, np.float64)-np.asarray(state['mean'],np.float64))/np.asarray(state['scale'],np.float64), -8, 8).astype(np.float32)
    return float(model.predict_proba(z[None])[0,1])


class RepairedGateTests(unittest.TestCase):
    def test_low_variance_cast_order_is_not_lost(self):
        values=np.array([.5000,.5001,.5002,.5003],np.float32)
        state=repaired_state({'kind':'ridge_logistic_v6','feature_names':['probe'],
            'mean':[float(np.mean(values,dtype=np.float64))],
            'scale':[float(np.std(values,dtype=np.float64))], 'coefficients':[1.], 'intercept':0.})
        olddiff=[]
        for value in values:
            x=np.asarray([value],np.float32)
            z=np.clip((x-np.asarray(state['mean'],np.float64))/np.asarray(state['scale'],np.float64),-8,8).astype(np.float32)
            reference=float(1/(1+np.exp(-float(z[0]))))
            with patch('optimized_video_gate_v6r.gate_features',return_value=(x,['probe'])):
                self.assertAlmostEqual(apply_gate(state,[],.5,{}),reference,places=14)
            with patch('optimized_video_gate_v6.gate_features',return_value=(x,['probe'])):
                olddiff.append(abs(old_apply(state,[],.5,{})-reference))
        self.assertGreater(max(olddiff),2e-6)

    def test_runtime_revision_is_opt_in(self):
        r=record(); fp=np.full(r['frames'],.7,np.float32); x,names=gate_features(fp,.7,r)
        state=repaired_state({'kind':'ridge_logistic_v6','feature_names':names,'mean':np.zeros(len(x)).tolist(),
                             'scale':np.ones(len(x)).tolist(),'coefficients':np.zeros(len(x)).tolist(),'intercept':-2.})
        bundle={'recipe':'unused','video_gate':state,'decoder':{'kind':'duration-logit-v4','threshold':.5,'transition_seconds':.1,'min_seconds':.12,'video_threshold':.4}}
        with patch('optimized_detector.predict_member',return_value=(fp,.7,None)):
            result=predict_record(bundle,r)
        self.assertEqual(result['intervals'],[])
        np.testing.assert_array_equal(result['frame_probabilities'],fp)
        self.assertAlmostEqual(result['video_probability'],1/(1+np.exp(2)),places=14)

    def test_invalid_revision_and_state_fail_closed(self):
        r=record(); fp=np.full(r['frames'],.7,np.float32); x,n=gate_features(fp,.5,r)
        state=repaired_state({'kind':'ridge_logistic_v6','feature_names':n,'mean':np.zeros(len(x)).tolist(),
                            'scale':np.ones(len(x)).tolist(),'coefficients':np.zeros(len(x)).tolist(),'intercept':0.})
        state['scale'][0]=0
        with self.assertRaises(ValueError):apply_gate(state,fp,.5,r)
        with self.assertRaises(ValueError):apply_gate({},fp,.5,r)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'WSL sklearn runtime required')
    def test_fitted_logistic_matches_independent_sklearn(self):
        records=[record(i) for i in range(12)]
        for i,r in enumerate(records): r['sha256']=str(i); r['labels'][:]=0 if i%3==0 else 1
        fp={i:np.linspace(.3+i*.01,.6+i*.01,r['frames'],dtype=np.float32) for i,r in enumerate(records)}
        vp={i:.4+i*.02 for i in fp}
        state=fit_gate('ridge_logistic_v6',records,list(fp),fp,vp,101)
        for i,r in enumerate(records):
            x,_=gate_features(fp[i],vp[i],r)
            self.assertAlmostEqual(apply_gate(state,fp[i],vp[i],r),sklearn_reference(state,x),places=13)

if __name__=='__main__': unittest.main()
