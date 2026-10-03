"""Production portable compatibility for opt-in experimental bundles only."""
import copy
import unittest
import numpy as np
from optimized_feature_blocks_v10 import KIND, RECIPES, feature_view_blocks
from optimized_compact_features_v4 import feature_view_v4, fit_rgb_pca
from optimized_detector import predict_record, required_channels
from optimized_proposal_review_v11 import KIND as REVIEW_KIND
from test_optimized_compact_v4 import record


def member(recipe,r,prob,kind=KIND,pca=None):
    x,v,names=(feature_view_blocks if kind==KIND else feature_view_v4)(recipe,r,pca)
    return {'recipe':recipe,'transform':{'feature_view':kind,'pca':pca,'frame_feature_names':names},
        'frame_model':{'kind':'constant','n_features':x.shape[1],'probability':prob},
        'video_model':{'kind':'constant','n_features':len(v),'probability':.9},'weight':1.}


def bundle(m):
    return {'members':[m],'recipe':m['recipe'],'calibration':None,'decoder':{
        'kind':'recall-stable-v2','threshold':.5,'low_ratio':.7,'smooth_seconds':.15,
        'gap_seconds':.12,'min_seconds':.12,'seed_seconds':.05,'video_threshold':0.,'video_strength':0.,'video_floor':0.}}


class ExperimentalRuntime(unittest.TestCase):
    def test_block_channels_and_missing_omitted_data(self):
        rs=[record(i) for i in range(3)];pca=fit_rgb_pca(rs,[0,1])
        for recipe in RECIPES:
            r=copy.deepcopy(rs[2]);b=bundle(member(recipe,r,.7,pca=pca if 'rgb' in recipe else None))
            expect={'motion','local'}
            if 'corrected' in recipe:expect.add('corrected')
            else:r.pop('corrected');r.pop('corrected_names')
            if 'rgb' in recipe:expect.add('rgb')
            else:r.pop('rgb')
            self.assertEqual(required_channels(b),expect)
            pred=predict_record(b,r);self.assertEqual(pred['intervals'],[(0,59)])

    def test_verifier_filters_spans_not_frame_or_video_scores(self):
        r=record();b=bundle(member(RECIPES[0],r,.7));a=predict_record(b,r)
        b['proposal_verifier']={'schema_version':'proposal-verifier-v11',
            'member':member('compact_motion_hgb',r,.1,kind='compact-fps-context-v4'),
            'review':{'kind':REVIEW_KIND,'minimum':.45,'rescue_peak':None}}
        c=predict_record(b,r)
        self.assertEqual(c['intervals'],[])
        for k in ['frame_probabilities','score']:np.testing.assert_array_equal(a[k],c[k])
        self.assertEqual(a['video_probability'],c['video_probability'])
        self.assertFalse(c['proposal_review_trace'][0]['accepted'])
        b['proposal_verifier']['review']['minimum']=0
        self.assertEqual(predict_record(b,r)['intervals'],a['intervals'])

    def test_verifier_channels_union_includes_reviewer(self):
        rs=[record(i) for i in range(3)];r=rs[2];pca=fit_rgb_pca(rs,[0,1])
        b=bundle(member(RECIPES[0],r,.7))
        b['proposal_verifier']={'schema_version':'proposal-verifier-v11',
            'member':member('compact_rgb_corrected_motion_hgb',r,.6,kind='compact-fps-context-v4',pca=pca),
            'review':{'kind':REVIEW_KIND,'minimum':.45,'rescue_peak':None}}
        self.assertEqual(required_channels(b),{'motion','local','corrected','rgb'})

    def test_nonleaf_and_unverified_combinations_fail_closed(self):
        r=record();original=bundle(member(RECIPES[0],r,.7))
        original['proposal_verifier']={'schema_version':'proposal-verifier-v11',
            'member':member('compact_motion_hgb',r,.6,kind='compact-fps-context-v4'),
            'review':{'kind':REVIEW_KIND,'minimum':.45,'rescue_peak':None}}
        mutations=[lambda b:b['proposal_verifier']['member'].update(members=[]),
            lambda b:b['proposal_verifier']['member'].update(boundary_models={}),
            lambda b:b.update(calibration={'kind':'identity'}),
            lambda b:b.update(video_gate={}),
            lambda b:b['decoder'].update(boundary_seconds=.2),
            lambda b:b['members'][0].update(boundary_models={}),
            lambda b:b['proposal_verifier']['review'].update(minimum=float('nan'))]
        for mutate in mutations:
            b=copy.deepcopy(original);mutate(b)
            with self.assertRaises(ValueError):required_channels(b)
            with self.assertRaises(ValueError):predict_record(b,r)

    def test_cache_only_spatial_experiment_is_not_live_deployable(self):
        for view in ['spatial-jepa-v13','spatial-jepa-mask-repair-v13r']:
            with self.assertRaisesRegex(ValueError,'no verified live extraction'):
                required_channels({'recipe':'spatial_jepa_corrected_motion_et',
                    'transform':{'feature_view':view}})

    def test_unknown_verifier_fails_closed(self):
        r=record();b=bundle(member(RECIPES[0],r,.7));b['proposal_verifier']={'schema_version':'bad'}
        with self.assertRaises(ValueError):required_channels(b)
        with self.assertRaises(ValueError):predict_record(b,r)
        b['proposal_verifier']=None
        self.assertNotIn('proposal_review_trace',predict_record(b,r))

if __name__=='__main__':unittest.main()
