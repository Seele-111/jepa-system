"""Focused portable model, ensemble and public-demo dispatch regression."""
import io,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from optimized_detector import required_channels,predict_record,learned_segments
from optimized_feature_view import feature_view
from optimized_calibration_state import validate_calibration_state
from optimized_locator import load_bundle
from optimized_probability_calibration import apply_calibration
from optimized_joint_calibration import apply_joint, fit_joint
import demo_detector as demo
import demo_app as web

class PortableDetectorTests(unittest.TestCase):
    def setUp(self):
        self.r={'fps':10.,'motion':np.arange(6,dtype=np.float32)[:,None],'motion_names':['u']}
        x,v,n=feature_view('motion_rf',self.r)
        self.model={'recipe':'motion_rf','transform':{'pca':None,'frame_feature_names':n},
                    'frame_model':{'kind':'constant','n_features':x.shape[1],'probability':.8},
                    'video_model':{'kind':'constant','n_features':len(v),'probability':.25},
                    'decoder':{'threshold':.6,'low_ratio':1,'smooth_seconds':0,'video_threshold':0}}
    def test_allowed_requirements_and_legacy_rejection(self):
        self.assertEqual(required_channels({'recipe':'rgb_corrected_motion_rf'}),{'rgb','corrected','motion'})
        for name in ['jepa_rf','hybrid_rf']:
            with self.subTest(name=name),self.assertRaises(ValueError):required_channels({'recipe':name})
    def test_inclusive_segment_and_exclusive_seconds(self):
        p=predict_record(self.model,self.r);self.assertEqual(p['intervals'],[(0,5)])
        segment=learned_segments(p,10)[0];self.assertEqual(segment['end_seconds'],.6);self.assertEqual(segment['frame_count'],6)
    def test_video_modulation_keeps_raw_scores_separate(self):
        self.model['decoder']['video_strength']=.5
        p=predict_record(self.model,self.r);self.assertEqual(p['intervals'],[])
        np.testing.assert_allclose(p['score'],.4);np.testing.assert_allclose(p['frame_probabilities'],.8)
    def test_hard_gate_can_return_empty_without_normal_guarantee(self):
        self.model['decoder']['video_threshold']=.5
        self.assertEqual(predict_record(self.model,self.r)['intervals'],[])
    def test_feature_order_mismatch_is_not_silent(self):
        self.model['transform']['frame_feature_names'][0]='wrong'
        with self.assertRaises(ValueError):predict_record(self.model,self.r)
    def test_ensemble_weighted_probabilities(self):
        from copy import deepcopy
        a=deepcopy(self.model);a['weight']=.25;a['frame_model']['probability']=.4
        b=deepcopy(self.model);b['weight']=.75
        result=predict_record({'recipe':'ensemble','members':[a,b],'decoder':self.model['decoder']},self.r)
        np.testing.assert_allclose(result['score'],.7);self.assertEqual(result['intervals'],[(0,5)])
    def test_invalid_ensemble_weights_rejected(self):
        self.model['weight']=-1
        with self.assertRaises(ValueError):predict_record({'members':[self.model],'decoder':self.model['decoder']},self.r)
    def test_refinement_requires_boundary_evidence(self):
        self.model['decoder']['boundary_seconds']=.2
        with self.assertRaises(ValueError):predict_record(self.model,self.r)

    def test_explicit_none_calibration_is_exactly_the_legacy_absent_path(self):
        expected = predict_record(self.model, self.r)
        self.model['calibration'] = None
        actual = predict_record(self.model, self.r)
        for name in ('frame_probabilities', 'score'):
            np.testing.assert_array_equal(actual[name], expected[name])
        self.assertEqual(actual['video_probability'], expected['video_probability'])
        self.assertEqual(actual['intervals'], expected['intervals'])

    def test_separate_calibration_dispatches_both_existing_branches(self):
        state = {'frame': {'kind': 'logit_affine_v1', 'slope': 1.2, 'intercept': -.3},
                 'video': {'kind': 'logit_affine_v1', 'slope': .8, 'intercept': .2}}
        self.model['calibration'] = state
        result = predict_record(self.model, self.r)
        expected = apply_calibration(np.full(6, .8, np.float32), state['frame'])
        video = float(apply_calibration([.25], state['video'])[0])
        np.testing.assert_array_equal(result['frame_probabilities'], expected)
        np.testing.assert_array_equal(result['score'], expected)
        self.assertEqual(result['video_probability'], video)

    def test_joint_calibration_changes_frames_but_not_video_probability(self):
        state = {'kind': 'monotone_joint_logit_v1', 'slope': 1.3,
                 'video_slope': .7, 'intercept': -.2}
        self.model['calibration'] = state
        result = predict_record(self.model, self.r)
        np.testing.assert_array_equal(result['frame_probabilities'],
                                      apply_joint(np.full(6, .8, np.float32), .25, state))
        self.assertEqual(result['video_probability'], .25)

    def test_joint_single_class_identity_fallback_is_legal(self):
        expected = predict_record(self.model, self.r)
        self.model['calibration'] = fit_joint([np.array([.2, .8])], [.25], [np.zeros(2)])
        self.assertEqual(self.model['calibration']['kind'], 'identity')
        actual = predict_record(self.model, self.r)
        np.testing.assert_array_equal(actual['frame_probabilities'], expected['frame_probabilities'])
        self.assertEqual(actual['video_probability'], expected['video_probability'])
        self.assertEqual(actual['intervals'], expected['intervals'])

    def test_invalid_calibration_fails_before_member_prediction(self):
        states = [{}, [], False, 0, '', {'kind': 'other'},
                  {'kind': 'other', 'frame': None, 'video': None},
                  {'frame': None}, {'frame': None, 'video': None, 'unknown': 1},
                  {'frame': {}, 'video': None},
                  {'frame': {'kind': 'identity', 'unknown': 0}, 'video': None},
                  {'frame': {'kind': 'logit_affine_v1', 'slope': 1.}, 'video': None},
                  {'kind': 'monotone_joint_logit_v1', 'slope': 1., 'intercept': 0.},
                  {'kind': 'monotone_joint_logit_v1', 'slope': np.inf,
                   'video_slope': 0., 'intercept': 0.},
                  {'kind': 'identity', 'slope': 1., 'video_slope': 0., 'intercept': np.nan}]
        for state in states:
            with self.subTest(state=state), patch('optimized_detector.predict_member') as member:
                self.model['calibration'] = state
                with self.assertRaisesRegex(ValueError, 'calibration'):
                    predict_record(self.model, self.r)
                member.assert_not_called()

    def test_ensemble_rejects_unknown_top_kind_before_averaging(self):
        member = dict(self.model, weight=1.)
        bundle = {'members': [member], 'decoder': self.model['decoder'],
                  'calibration': {'kind': 'future_monotone_stack_v3'}}
        with patch('optimized_detector.predict_member') as predict:
            with self.assertRaisesRegex(ValueError, 'calibration'):
                predict_record(bundle, self.r)
        predict.assert_not_called()

    def test_invalid_loaded_state_fails_before_video_io_or_feature_extraction(self):
        from optimized_detector import analyze_optimized_video
        for state in ({'kind': 'other'}, {}, {'frame': None},
                      {'frame': None, 'video': None, 'unknown': 1}):
            bundle = dict(self.model, calibration=state)
            with self.subTest(state=state), \
                    patch('optimized_detector.load_bundle', return_value=bundle) as load, \
                    patch('optimized_detector.Path.is_file', return_value=True), \
                    patch('optimized_detector.Path.mkdir'), \
                    patch.object(demo, '_video_info') as info, \
                    patch('optimized_detector.extract_video_features') as extract, \
                    patch('optimized_detector._request_features') as worker:
                with self.assertRaisesRegex(ValueError, 'calibration'):
                    analyze_optimized_video('not-read.mp4', 'not-written', bundle_path='not-read.json')
                load.assert_called_once()
                info.assert_not_called()
                extract.assert_not_called()
                worker.assert_not_called()

class DemoDispatchTests(unittest.TestCase):
    def test_optimized_dispatch_does_not_enter_legacy_or_read_labels(self):
        with patch('optimized_detector.analyze_optimized_video',return_value={'algorithm':'optimized'}) as optimized,patch.object(demo,'_run_wsl_true_jepa') as legacy:
            result=demo.analyze_video('v.mp4','out',algorithm='optimized');self.assertEqual(result['algorithm'],'optimized');optimized.assert_called_once();legacy.assert_not_called()
    def test_fast_dispatch_preserves_mode_and_uses_only_fast_bundle(self):
        with patch('optimized_detector.analyze_optimized_video',return_value={'algorithm':'optimized_fast'}) as optimized,patch.object(demo,'_run_wsl_true_jepa') as legacy:
            result=demo.analyze_video('v.mp4','out',algorithm='optimized_fast')
            self.assertEqual(result['algorithm'],'optimized_fast')
            self.assertEqual(optimized.call_args.kwargs['algorithm'],'optimized_fast')
            from jepa_runtime import settings
            self.assertEqual(optimized.call_args.kwargs['bundle_path'],settings().motion_bundle)
            legacy.assert_not_called()
    def test_invalid_public_mode_rejected_before_read(self):
        with self.assertRaises(demo.DemoDetectionError):demo.analyze_video('v.mp4','out',algorithm='unknown')
    def test_upload_invalid_algorithm_contract_is_400(self):
        with web.app.test_client() as client:
            response=client.post('/api/upload',data={'algorithm':'unknown','video':(io.BytesIO(b'not-video'),'x.mp4')},content_type='multipart/form-data')
        self.assertEqual(response.status_code,400)

class RealBundleCalibrationTests(unittest.TestCase):
    """Read installed portable JSONs; synthetic features keep the smoke test CPU-only."""
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        from jepa_runtime import settings
        cfg = settings()
        cls.bundles = [(path, load_bundle(path)) for path in (cfg.full_bundle, cfg.motion_bundle)]

    @staticmethod
    def record_for(bundle):
        record = {'fps': 10.}
        for channel, names in bundle['raw_feature_names'].items():
            record[channel] = np.zeros((8, len(names)), dtype=np.float32)
            record[channel + '_names'] = names
        for member in bundle.get('members') or [bundle]:
            pca = member['transform'].get('pca')
            if pca is not None:
                mean = np.asarray(pca['mean'], np.float32)
                record['rgb'] = np.broadcast_to(mean, (8, len(mean))).copy()
        return record

    def test_real_v1_bundles_load_and_none_matches_absent_exactly(self):
        for path, bundle in self.bundles:
            with self.subTest(model=path.name):
                self.assertEqual(validate_calibration_state(bundle.get('calibration')), 'identity')
                self.assertNotIn('calibration', bundle)
                record = self.record_for(bundle)
                expected = predict_record(bundle, record)
                actual = predict_record(dict(bundle, calibration=None), record)
                for name in ('frame_probabilities', 'score'):
                    np.testing.assert_array_equal(actual[name], expected[name])
                    self.assertEqual(actual[name].shape, (8,))
                    self.assertTrue(np.isfinite(actual[name]).all())
                self.assertEqual(actual['video_probability'], expected['video_probability'])
                self.assertEqual(actual['intervals'], expected['intervals'])

    def test_real_bundles_predict_with_existing_legal_calibration_schemas(self):
        affine = {'kind': 'logit_affine_v1', 'slope': 1.2, 'intercept': -.3}
        joint = {'kind': 'monotone_joint_logit_v1', 'slope': 1.3,
                 'video_slope': .7, 'intercept': -.2}
        identity = fit_joint([np.array([.2, .8])], [.4], [np.zeros(2)])
        states = [{'frame': deepcopy(affine), 'video': deepcopy(affine)},
                  {'frame': None, 'video': {'kind': 'identity', 'reason': 'single_class_inner_OOF'}},
                  joint, identity]
        for path, bundle in self.bundles:
            record = self.record_for(bundle)
            raw = predict_record(bundle, record)
            for state in states:
                with self.subTest(model=path.name, state=state):
                    actual = predict_record(dict(bundle, calibration=state), record)
                    frame, video = raw['frame_probabilities'], raw['video_probability']
                    if 'kind' in state:
                        expected_frame = apply_joint(frame, video, state)
                        expected_video = video
                    else:
                        expected_frame = apply_calibration(frame, state['frame'])
                        expected_video = float(apply_calibration([video], state['video'])[0])
                    np.testing.assert_array_equal(actual['frame_probabilities'], expected_frame)
                    self.assertEqual(actual['video_probability'], expected_video)
                    self.assertTrue(np.isfinite(actual['score']).all())


class StackDispatchTests(unittest.TestCase):
    def fixture(self):
        from optimized_stack_v3 import FAST_BRANCHES,ANCHOR
        state={'kind':'monotone_stack_logit_v3','branches':FAST_BRANCHES,'coefficients':ANCHOR.tolist(),'ridge':1.}
        members=[{'recipe':b[0][0],'weight':1/3} for b in FAST_BRANCHES]
        bundle={'members':members,'calibration':state,'decoder':{'kind':'recall-stable-v2','threshold':.4}}
        outputs=[(np.full(6,p,np.float32),v,None) for p,v in [(.2,.25),(.6,.5),(.8,.75)]]
        return bundle,outputs
    def test_stack_matches_member_specific_numpy_reference_not_average(self):
        from optimized_stack_v3 import apply_stack
        from unittest.mock import patch
        bundle,outputs=self.fixture()
        frames={m['recipe']:o[0] for m,o in zip(bundle['members'],outputs)}
        videos={m['recipe']:o[1] for m,o in zip(bundle['members'],outputs)}
        expected,video=apply_stack(frames,videos,bundle['calibration'])
        with patch('optimized_detector.predict_member',side_effect=outputs):out=predict_record(bundle,{'fps':10.})
        np.testing.assert_array_equal(out['frame_probabilities'],expected)
        self.assertEqual(out['video_probability'],video);self.assertEqual(out['intervals'],[(0,5)])
    def test_invalid_stack_member_profile_rejected_before_member_inference(self):
        from unittest.mock import patch
        from copy import deepcopy
        bundle,outputs=self.fixture()
        for names in [['motion_rf','motion_rf','local_motion_et'],['motion_rf','motion_et','typo'],[]]:
            bad=deepcopy(bundle);bad['members']=[{'recipe':r,'weight':1/3} for r in names]
            with self.subTest(names=names),patch('optimized_detector.predict_member') as infer:
                with self.assertRaises(ValueError):predict_record(bad,{'fps':10.})
                infer.assert_not_called()
    def test_stack_dispatch_refuses_to_guess_original_members(self):
        from optimized_calibration_state import apply_calibration_state
        bundle,outputs=self.fixture()
        with self.assertRaises(ValueError):apply_calibration_state(np.full(6,.5),.5,bundle['calibration'])


if __name__=='__main__':unittest.main()
