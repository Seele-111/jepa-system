"""Independent golden cases and substantive tampering; never fit any model."""
from __future__ import annotations
from copy import deepcopy
import contextlib
import gzip
from io import BytesIO,StringIO
import json
from pathlib import Path
import pickle
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import audit_baseline_interval_quality as I

def records(n=6):
    return [{'name':str(i),'sha256':f'{i+1:064x}','frames':8,'fps':10.,'event_count':i%2,
             'generator':'g','labels':np.array([0,1,1,0,0,0,0,0] if i%2 else [0]*8,bool)} for i in range(n)]

def leaf(value=.8):return {'left':[-1],'right':[-1],'feature':[-2],'threshold':[-2.],'value':[value]}

def model():return {'kind':'interval-quality-et-v1','n_features':1,'trees':[leaf() for _ in range(192)]}

def resign(row):
    row.pop('metadata_sha256',None);row['metadata_sha256']=I.A.value_hash(I.producer_row(row));return row

class Partitions(unittest.TestCase):
    def test_alias_SHA_cannot_cross_fit_predict_or_excluded(self):
        rs=records();rs[1]['sha256']=rs[0]['sha256']
        for a,b,c in (([0],[1],[]),([0],[2],[1]),([2],[0],[1])):
            with self.subTest(a=a,b=b,c=c),self.assertRaises(I.A.AuditError):I.partition(rs,a,b,c)
    def test_overlap_duplicate_bool_and_outside_rows_rejected(self):
        rs=records()
        for a,b,c in (([0,0],[1],[]),([False],[1],[]),([6],[1],[]),([0],[0],[])):
            with self.assertRaises(I.A.AuditError):I.partition(rs,a,b,c)
    def test_fullfit_only_allowed_explicitly_and_with_same_ids(self):
        rs=records();I.partition(rs,[0,1],[0,1],full=True)
        with self.assertRaises(I.A.AuditError):I.partition(rs,[0,1],[1,0],full=True)
    def test_deep_jobs_reconstruct_exclusions_and_final_coverage(self):
        rs=records(18);rs.append({**rs[1],'name':'alias'})
        outer=I.A.seeded_folds(rs,I.A.content_groups(rs,list(range(len(rs)))),5,20261002)
        jobs=I.expected_jobs(rs,outer);self.assertEqual(len(jobs),57)
        self.assertEqual(sum(j['scope']<5 for j in jobs),45)
        for j in jobs:
            I.partition(rs,j['fit'],j['predict'],j['excluded'])
            if j['scope']<5:self.assertTrue(set(outer[j['scope']])<=set(j['excluded']))
    def test_merge_oof_missing_extra_or_duplicate_fails(self):
        a=I.A.Probabilities({0:np.zeros(8,np.float32)},{0:.5})
        for parts,ids in (([a,a],[0]),([a],[0,1]),([a],[])):
            with self.assertRaises(I.A.AuditError):I.merge_scores(parts,ids)

class Probabilities(unittest.TestCase):
    def test_shapes_values_dtypes_and_extra_keys_fail(self):
        rs=records();good={'frame_0':np.full(8,.5,np.float32),'video_0':np.array(.5)}
        I.validate_scores(good,rs,[0])
        for bad in ({**good,'frame_0':np.full(7,.5,np.float32)}, {**good,'frame_0':np.full(8,.5,np.float64)},
            {**good,'video_0':np.array([.5])},{**good,'video_0':np.array(np.nan)},
            {**good,'frame_0':np.full(8,1.01,np.float32)},{**good,'video_1':np.array(.5)}):
            with self.assertRaises(I.A.AuditError):I.validate_scores(bad,rs,[0])
    def test_hash_checked_before_unpickler(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'state.gz';p.write_bytes(b'not gzip')
            with patch.object(I.LocalStateUnpickler,'load',side_effect=AssertionError('executed')) as load:
                with self.assertRaisesRegex(I.A.AuditError,'SHA256'):I.read_state(I.A.Evidence(t),'state.gz','f'*64)
                load.assert_not_called()
    def test_decompressed_state_bound_checked_before_load(self):
        with tempfile.TemporaryDirectory() as t:
            data=gzip.compress(b'x'*100);(Path(t)/'state.gz').write_bytes(data)
            with patch.object(I,'MAX_STATE_BYTES',16),patch.object(I.LocalStateUnpickler,'load') as load:
                with self.assertRaisesRegex(I.A.AuditError,'bound'):I.read_state(I.A.Evidence(t),'state.gz',I.A.sha(data))
                load.assert_not_called()
    def test_unapproved_pickle_global_never_executes(self):
        class Bad:
            def __reduce__(self):return eval,('1+1',)
        payload=pickle.dumps({r:Bad() for r,w in I.MEMBERS})
        with tempfile.TemporaryDirectory() as t:
            data=gzip.compress(payload);(Path(t)/'state.gz').write_bytes(data)
            with self.assertRaisesRegex(I.A.AuditError,'unapproved pickle'):I.read_state(I.A.Evidence(t),'state.gz',I.A.sha(data))
    def test_plain_trusted_state_and_trailing_bytes(self):
        state={r:{} for r,w in I.MEMBERS}
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'state.gz';data=gzip.compress(pickle.dumps(state));p.write_bytes(data)
            self.assertEqual(I.read_state(I.A.Evidence(t),'state.gz',I.A.sha(data)),state)
            data=gzip.compress(pickle.dumps(state)+b'junk');p.write_bytes(data)
            with self.assertRaisesRegex(I.A.AuditError,'trailing'):I.read_state(I.A.Evidence(t),'state.gz',I.A.sha(data))

class TargetsAndSelection(unittest.TestCase):
    def test_inclusive_max_iou_multi_event_and_normal_goldens(self):
        y=np.array([0,1,1,0,1,1,0,0],bool)
        actual=I.independent_targets([(1,2),(2,4),(0,7),(7,7)],y)
        np.testing.assert_array_equal(actual,np.array([1,.25,.25,0],np.float32))
        np.testing.assert_array_equal(I.independent_targets([(0,7)],np.zeros(8,bool)),[0.])
    def test_sampler_caps_seed_reproducibility_and_strata(self):
        y=np.r_[np.zeros(80),np.full(80,.1),np.full(80,.5)].astype(np.float32)
        a=I.sample_ids(y,False,123);self.assertEqual(len(a),96)
        self.assertEqual([sum((a>=j*80)&(a<(j+1)*80)) for j in range(3)],[32]*3)
        np.testing.assert_array_equal(a,I.sample_ids(y,False,123))
        self.assertFalse(np.array_equal(a,I.sample_ids(y,False,124)))
        self.assertEqual(len(I.sample_ids(np.zeros(200,np.float32),True,123)),96)
    def test_weights_quality_strata_content_mass_and_normal_two(self):
        y=np.array([0,.05,.1,.3,.5,.9],np.float32);w=I.independent_weights(y,False,.5)
        np.testing.assert_array_equal(w,np.full(6,.25,np.float32))
        self.assertEqual(float(I.independent_weights(np.zeros(8,np.float32),True,.5).sum()),1.)
    def test_empty_candidates_targets_sampler_weights(self):
        self.assertEqual(I.independent_targets([],np.zeros(8,bool)).shape,(0,))
        self.assertEqual(I.sample_ids(np.zeros(0,np.float32),False,1).shape,(0,))
        self.assertEqual(I.independent_weights(np.zeros(0,np.float32),True,1).shape,(0,))
    def test_selector_tie_order_and_inclusive_overlap(self):
        self.assertEqual(I.select_intervals([(2,3),(1,2),(4,5)],np.full(3,.5,np.float32),.5,8),[(1,2),(4,5)])
    def test_selector_bad_values_and_outside_span(self):
        for b,s in (([(1,9)],np.array([.5])), ([(1,2)],np.array([np.nan])), ([(1,2)],np.array([1.1]))):
            with self.assertRaises(I.A.AuditError):I.select_intervals(b,s,.5,8)
    def test_greedy_matching_order_and_frame_union(self):
        y=np.array([1,1,0,1,1,0,0,0],bool)
        s=I.A.row_statistics([(0,3),(0,1)],y)
        self.assertEqual(tuple(s[:6]),(1,1,1,1,1,1));self.assertEqual(tuple(s[6:9]),(3,1,1))
    def test_utility_golden_project_coefficients_not_v8(self):
        stats=np.array([1,0,0,1,0,0,0,0,0,1,1,0])
        self.assertEqual(I.utility(stats,1,1),.65)
    def test_five_configs_top3_tie_order_and_q20_golden(self):
        rs=records();items={i:{'intervals':np.array([[1,2]],np.int32),'scores':np.array([.4 if i%2 else .2],np.float32),'frame':np.full(8,.1,np.float32),'video':.9} for i in range(6)}
        decoder={'threshold':.5,'min_seconds':0,'smooth_seconds':0,'gap_seconds':0,'video_threshold':0}
        chosen=I.choose_independent(rs,list(items),items,123,decoder)
        self.assertEqual(chosen['ordinal'],1);self.assertEqual(chosen['stable_utility'],1.)
        self.assertEqual(len(chosen['all_configurations']),5)
        self.assertEqual([r['ordinal'] for r in chosen['all_configurations'] if 'bootstrap_q20' in r],[0,1,2])
        changed=deepcopy(chosen);changed['ordinal']=2
        with self.assertRaises(I.A.AuditError):I.compare(changed,chosen,'tampered selection')
    def test_paired_differences_are_paired_no_refit(self):
        rs=records();old={i:[] for i in range(6)};new={i:[(1,2)] if i%2 else [] for i in range(6)}
        d=I.paired_differences(rs,old,new)['deltas']
        self.assertEqual(d['F1_IoU05']['primary_minus_control'],1.)
        self.assertEqual(d['positive_empty_rows']['primary_minus_control'],-3.)

class Portable(unittest.TestCase):
    def test_threshold_equality_leaf_values_and_empty_validation(self):
        m={'kind':'interval-quality-et-v1','n_features':1,'trees':[{'left':[1,-1,-1],'right':[2,-1,-1],'feature':[0,-2,-2],'threshold':[.5,-2,-2],'value':[.5,.2,.8]}]}
        np.testing.assert_array_equal(I.portable_head(m,np.array([[.5],[.5001]],np.float32),1),np.array([.2,.8],np.float32))
        self.assertEqual(I.portable_head(m,np.empty((0,1),np.float32),1).shape,(0,))
    def test_cycle_shared_unreachable_bad_value_and_index_rejected(self):
        base={'kind':'interval-quality-et-v1','n_features':1,'trees':[leaf()]}
        bads=[]
        for field,value in (('left',[0]),('value',[np.nan]),('feature',[-1])):
            b=deepcopy(base);b['trees'][0][field]=value;bads.append(b)
        b=deepcopy(base);b['trees'][0]={k:v+v for k,v in b['trees'][0].items()};bads.append(b)
        for b in bads:
            with self.assertRaises(I.A.AuditError):I.portable_head(b,np.empty((0,1),np.float32),1)
    def test_tree_count_dtype_and_width_must_match(self):
        for x in (np.zeros((2,2),np.float32),np.zeros((2,1),np.float64)):
            with self.assertRaises(I.A.AuditError):I.portable_head(model(),x)
        m=model();m['trees'].pop()
        with self.assertRaises(I.A.AuditError):I.portable_head(m,np.zeros((2,1),np.float32))

class ProducerAndRepair(unittest.TestCase):
    def test_integer_key_hash_sorting_1_2_10_not_lexicographic(self):
        r={'predict':[1,2,10],'prediction_candidate_counts':{'1':1,'2':2,'10':3}};resign(r);I.signed_head(r)
        with self.assertRaises(I.A.AuditError):I.signed(r,'metadata_sha256')
    def test_noncanonical_keys_values_and_coverage_rejected(self):
        for c in ({'01':1},{'+1':1},{'1':True},{'1':4097},{'1':-1},{'2':1},{'1':1.0}):
            with self.assertRaises(I.A.AuditError):I.producer_row({'predict':[1],'prediction_candidate_counts':c})
    def test_rekey_cannot_hide_partition_or_feature_digest_tampering(self):
        r={'predict':[1],'prediction_candidate_counts':{'1':1},'features_sha256':'a'*64};resign(r)
        for k,v in (('predict',[2]),('features_sha256','b'*64)):
            b=deepcopy(r);b[k]=v
            with self.assertRaises(I.A.AuditError):I.signed_head(b)
    def test_exact_repair_ast_and_unrelated_change_rejected(self):
        parent=(I.ROOT/'code/run_baseline_interval_quality.py').read_text('utf-8');repair=(I.ROOT/'code/run_baseline_interval_quality_repaired.py').read_text('utf-8')
        self.assertTrue(I.repair_source_equal(parent,repair)['parent_repaired_AST_equal_after_exact_allowed_edits'])
        bad=repair.replace('THRESHOLDS=(.25,.35,.45,.55)','THRESHOLDS=(.25,.35,.45,.65)')
        self.assertNotEqual(bad,repair)
        with self.assertRaises(I.A.AuditError):I.repair_source_equal(parent,bad)
    def test_normalization_elsewhere_is_not_authorized(self):
        parent=(I.ROOT/'code/run_baseline_interval_quality.py').read_text('utf-8');repair=(I.ROOT/'code/run_baseline_interval_quality_repaired.py').read_text('utf-8')
        with self.assertRaises(I.A.AuditError):I.repair_source_equal(parent,repair.replace("list(names)!=protocol['interval_feature_names']","tuple(names)!=protocol['interval_feature_names']"))

class HeadTampering(unittest.TestCase):
    def fixture(self,t):
        rs=records(2);rs[0]['labels']=rs[1]['labels'].copy();rs[0]['event_count']=1
        p=I.A.Probabilities({0:np.full(8,.5,np.float32)},{0:.4});q=I.A.Probabilities({1:np.full(8,.6,np.float32)},{1:.4})
        fs=['deep0','deep1','deep2'];ps=['legacy'];inputs=(p,q,[0],[1],fs,ps);seed=20278000
        Q=types.SimpleNamespace(proposal_bank=lambda *_:[(1,2)],interval_features=lambda *_:(np.array([[.25]],np.float32),('x',)))
        row={'status':'complete','scope':0,'inner':0,'full_fit':False,'seed':seed,'fit':[0],'predict':[1],
         'fit_content_sha256':[rs[0]['sha256']],'predict_content_sha256':[rs[1]['sha256']],
         'fit_base_metadata':fs,'predict_base_metadata':ps,'baseline_protocol_sha256':I.BASE_RECEIPT,'interval_protocol_sha256':I.HEAD_RECEIPT,
         'sampling':[{'index':0,'candidate_count':1,'selected_ids':[0],'seed':seed,'features_sha256':I.A.sha(np.array([[.25]],np.float32).tobytes()),'targets_sha256':I.A.sha(np.array([1],np.float32).tobytes()),'weight_mass':1.}],
         'training_proposal_rows':1,'prediction_candidate_counts':{'1':1},'portable_max_error':0.}
        base=Path(t)/I.HEAD/'fits/s0_inner0';base.parent.mkdir(parents=True)
        state={'schema_version':'interval-quality-state-v1','model':model(),'interval_feature_names':['x'],'raw_evidence_names':['raw'],'fit_seed':seed,'fit_content_sha256':[rs[0]['sha256']]}
        base.with_suffix('.model.json').write_text(json.dumps(state),'utf-8');row['model_sha256']=I.A.sha(base.with_suffix('.model.json').read_bytes())
        a={'intervals_1':np.array([[1,2]],np.int32),'scores_1':np.array([.8],np.float32),'frame_1':q.frame[1],'video_1':np.array(.4)}
        with base.with_suffix('.npz').open('wb') as f:np.savez_compressed(f,**a)
        row['npz_sha256']=I.A.sha(base.with_suffix('.npz').read_bytes());resign(row)
        base.with_suffix('.json').write_text(json.dumps(row),'utf-8')
        return rs,inputs,Q,base,row,a
    def check(self,t,rs,inputs,Q):
        with patch.object(I,'head_inputs',return_value=inputs):
            return I.load_head(I.A.Evidence(t),{'interval_feature_names':['x']},rs,[],{}, {},{i:np.zeros((8,1)) for i in range(len(rs))},['raw'],Q,0,0)
    def test_usable_fixture_replays_all_heldout_scores(self):
        with tempfile.TemporaryDirectory() as t:
            rs,inputs,Q,base,row,a=self.fixture(t);items,proof=self.check(t,rs,inputs,Q)
            self.assertEqual(set(items),{1});self.assertEqual(proof['portable_replay_max_abs_error'],0.)
    def test_sparse_global_fit_ids_and_alias_weights_are_not_positional(self):
        with tempfile.TemporaryDirectory() as t:
            _,_,Q,base,row,a=self.fixture(t);rs=records(10);train=[9,7]
            rs[7]['sha256']=rs[9]['sha256']
            fit=I.A.Probabilities({i:np.full(8,.5,np.float32) for i in train},{i:.4 for i in train})
            pred=I.A.Probabilities({1:np.full(8,.6,np.float32)},{1:.4})
            inputs=(fit,pred,train,[1],['deep0','deep1','deep2'],['legacy'])
            row['fit']=train;row['fit_content_sha256']=[rs[9]['sha256']]
            template=deepcopy(row['sampling'][0]);row['sampling']=[]
            for i in train:
                row['sampling'].append({**template,'index':i,'seed':20278000+10007*i,'weight_mass':.5})
            row['training_proposal_rows']=2
            state=json.loads(base.with_suffix('.model.json').read_text('utf-8'))
            state['fit_content_sha256']=row['fit_content_sha256']
            base.with_suffix('.model.json').write_text(json.dumps(state),'utf-8')
            row['model_sha256']=I.A.sha(base.with_suffix('.model.json').read_bytes())
            resign(row);base.with_suffix('.json').write_text(json.dumps(row),'utf-8')
            items,proof=self.check(t,rs,inputs,Q)
            self.assertEqual(proof['fit'],train);self.assertEqual(proof['sample_rows'],2)
            self.assertEqual(set(items),{1});self.assertEqual(proof['portable_replay_max_abs_error'],0.)
            row['sampling'][0]['weight_mass']=1.;resign(row)
            base.with_suffix('.json').write_text(json.dumps(row),'utf-8')
            with self.assertRaisesRegex(I.A.AuditError,'sampling/weight'):
                self.check(t,rs,inputs,Q)
    def test_resigned_metadata_wrong_sources_sample_seed_digest_weights_fail(self):
        changes=[('fit_base_metadata',['bad']),('predict_base_metadata',['bad']),('seed',1),('fit_content_sha256',['f'*64]),('training_proposal_rows',2)]
        for key,value in changes:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as t:
                rs,inputs,Q,base,row,a=self.fixture(t);row[key]=value;resign(row);base.with_suffix('.json').write_text(json.dumps(row),'utf-8')
                with self.assertRaises(I.A.AuditError):self.check(t,rs,inputs,Q)
        for key,value in (('selected_ids',[0,0]),('seed',1),('targets_sha256','f'*64),('features_sha256','f'*64),('weight_mass',2.)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as t:
                rs,inputs,Q,base,row,a=self.fixture(t);row['sampling'][0][key]=value;resign(row);base.with_suffix('.json').write_text(json.dumps(row),'utf-8')
                with self.assertRaises(I.A.AuditError):self.check(t,rs,inputs,Q)
    def test_resigned_npz_wrong_scores_base_provenance_candidate_fail(self):
        for key,value in (('scores_1',np.array([.7],np.float32)),('frame_1',np.full(8,.7,np.float32)),('intervals_1',np.array([[0,2]],np.int32)),('video_1',np.array(.9))):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as t:
                rs,inputs,Q,base,row,a=self.fixture(t);a[key]=value
                with base.with_suffix('.npz').open('wb') as f:np.savez_compressed(f,**a)
                row['npz_sha256']=I.A.sha(base.with_suffix('.npz').read_bytes());resign(row);base.with_suffix('.json').write_text(json.dumps(row),'utf-8')
                with self.assertRaises(I.A.AuditError):self.check(t,rs,inputs,Q)
    def test_reanchored_model_leaf_or_seed_does_not_pass(self):
        for mode in ('leaf','seed'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
                rs,inputs,Q,base,row,a=self.fixture(t);s=json.loads(base.with_suffix('.model.json').read_text('utf-8'))
                if mode=='leaf':s['model']['trees'][0]['value']=[.1]
                else:s['fit_seed']=1
                base.with_suffix('.model.json').write_text(json.dumps(s),'utf-8');row['model_sha256']=I.A.sha(base.with_suffix('.model.json').read_bytes());resign(row);base.with_suffix('.json').write_text(json.dumps(row),'utf-8')
                with self.assertRaises(I.A.AuditError):self.check(t,rs,inputs,Q)

class CLI(unittest.TestCase):
    def test_snapshot_pending_without_output_never_claims_pass(self):
        with patch.object(I,'run_audit',return_value={'status':'pending','deployment_approved':False}),contextlib.redirect_stdout(StringIO()) as out:
            self.assertEqual(I.main(['--snapshot']),2)
        self.assertEqual(json.loads(out.getvalue())['status'],'pending')
    def test_missing_evidence_fails_explicitly(self):
        with patch.object(I,'run_audit',side_effect=I.A.MissingArtifact('not ready')),contextlib.redirect_stdout(StringIO()) as out:
            self.assertEqual(I.main(['--snapshot']),1)
        self.assertEqual(json.loads(out.getvalue())['status'],'failed')
    def test_exclusive_output_and_fail_report(self):
        with tempfile.TemporaryDirectory() as t,patch.object(I,'ROOT',Path(t)):
            folder=Path(t)/I.REPAIR;folder.mkdir(parents=True);p=folder/'independent_audit.json';p.write_text('old','utf-8')
            with patch.object(I,'run_audit') as run,contextlib.redirect_stderr(StringIO()):
                self.assertEqual(I.main(['--output',str(p)]),2);run.assert_not_called()
            self.assertEqual(p.read_text('utf-8'),'old')
            p=folder/'independent_audit_failure.json'
            with patch.object(I,'run_audit',side_effect=I.A.AuditError('tampered selection')),contextlib.redirect_stdout(StringIO()):self.assertEqual(I.main(['--output',str(p)]),1)
            self.assertEqual(json.loads(p.read_text('utf-8'))['status'],'failed')

if __name__=='__main__':unittest.main()
