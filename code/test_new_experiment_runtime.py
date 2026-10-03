"""Compatibility checks for opt-in compact, boosted and duration/gate members."""
import copy,importlib.util,unittest
from unittest.mock import patch
import numpy as np
from optimized_detector import required_channels,predict_record
from optimized_duration_decoder_v4 import decode_duration
from optimized_event_training_v5 import fit_event_member,export_event_member,RECIPES
from optimized_compact_model_v4 import fit_compact_member,export_compact_member
from optimized_compact_features_v4 import RECIPES as COMPACT
from optimized_video_gate_v6 import gate_features
from test_optimized_compact_v4 import record

class ExperimentRuntime(unittest.TestCase):
    def test_compact_requirements(self):
        expected=[{'motion','local','corrected'},{'motion','local','corrected','rgb'},{'motion','local','corrected','rgb'},{'motion','local'}]
        for r,e in zip(COMPACT,expected):
            self.assertEqual(required_channels({'recipe':r,'transform':{'feature_view':'compact-fps-context-v4'}}),e)
        with self.assertRaises(ValueError):required_channels({'recipe':'unknown','transform':{'feature_view':'compact-fps-context-v4'}})

    def test_duration_branch_does_not_change_evidence(self):
        r=record();fp=np.full(r['frames'],.7,np.float32);cfg={'kind':'duration-logit-v4','threshold':.5,'transition_seconds':.1,'min_seconds':.12}
        bundle={'recipe':'unused','decoder':cfg}
        with patch('optimized_detector.predict_member',return_value=(fp,.2,None)):
            p=predict_record(bundle,r)
        np.testing.assert_array_equal(p['frame_probabilities'],fp);np.testing.assert_array_equal(p['score'],fp)
        self.assertEqual(p['intervals'],decode_duration(fp,r['fps'],.2,cfg))

    def test_video_gate_changes_only_video_evidence_and_gate(self):
        r=record();fp=np.full(r['frames'],.7,np.float32);x,n=gate_features(fp,.7,r)
        state={'kind':'ridge_logistic_v6','feature_names':n,'mean':np.zeros(len(x)).tolist(),'scale':np.ones(len(x)).tolist(),'coefficients':np.zeros(len(x)).tolist(),'intercept':-2.}
        cfg={'kind':'duration-logit-v4','threshold':.5,'transition_seconds':.1,'min_seconds':.12,'video_threshold':.4}
        bundle={'recipe':'unused','decoder':cfg,'video_gate':state}
        with patch('optimized_detector.predict_member',return_value=(fp,.7,None)):
            p=predict_record(bundle,r)
        self.assertEqual(p['intervals'],[]);self.assertAlmostEqual(p['video_probability'],1/(1+np.exp(2)))
        self.assertEqual(p['raw_video_probability'],.7);np.testing.assert_array_equal(p['frame_probabilities'],fp)
        bundle['video_gate']['kind']='other'
        with patch('optimized_detector.predict_member',return_value=(fp,.7,None)):
            with self.assertRaises(ValueError):predict_record(bundle,r)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'existing WSL training runtime required')
    def test_portable_new_members_pass_through_public_runtime(self):
        records=[record(i) for i in range(4)];records[0]['labels'][:]=0
        for kind,recipe in [('compact',COMPACT[2]),('event',RECIPES[2])]:
            fit=fit_compact_member if kind=='compact' else fit_event_member
            export=export_compact_member if kind=='compact' else export_event_member
            f,v,s=fit(recipe,records,[0,1,2],[3],42);member=export(s)
            b={'members':[member],'decoder':{'threshold':.5},'calibration':None};p=predict_record(b,records[3])
            np.testing.assert_allclose(p['frame_probabilities'],f[3],atol=2e-6,rtol=0);self.assertAlmostEqual(p['video_probability'],v[3],delta=2e-6)

if __name__=='__main__':unittest.main()
