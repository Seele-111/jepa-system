"""Independent goldens/tampering; no model fits and no sklearn needed for unit cases."""
from __future__ import annotations
from copy import deepcopy
import ast, contextlib, gzip, json, pickle, tempfile, types, unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import numpy as np
import audit_baseline_semantic_readout as S

def records(n=8):
    out=[]
    for i in range(n):
        y=np.array([0,1,1,0,0,0,0,0] if i%2 else [0]*8,bool)
        out.append(dict(name=str(i),generator='g',sha256=f'{i+1:064x}',frames=8,fps=10.,event_count=i%2,labels=y,
         semantic=np.zeros((8,16),np.float32),semantic_support=np.ones(8,bool),semantic_direct=np.ones(8,bool),
         motion=np.arange(16,dtype=np.float32).reshape(8,2),motion_names=['m0','m1'],corrected=np.arange(8,dtype=np.float32)[:,None],corrected_names=['c0']))
    return out

def pca(width=16,k=16,mean=None,components=None):
    return dict(kind=S.KIND,fit_role=S.ROLE,n_components=k,semantic_names=[f'sem/latent_{j:04d}' for j in range(width)],mean=[0.]*width if mean is None else mean,components=np.eye(width,dtype=np.float32)[:k].tolist() if components is None else components)

def leaf(value=.2):return dict(left=[-1],right=[-1],feature=[-2],threshold=[-2.],value=[value])
def forest(value=.2,width=1,trees=1):return dict(kind='forest',n_features=width,trees=[leaf(value) for _ in range(trees)])
def signed(row):
    row.pop('receipt_sha256',None);row['receipt_sha256']=S.A.value_hash(row);return row

def fake_native(algorithm='rf',seed=17,video=False,width=1):
    rf=algorithm=='rf' and not video;n=128 if video else 160 if rf else 192
    params=dict(n_estimators=n,max_depth=4 if video else 8 if rf else 9,min_samples_leaf=2 if video else 10 if rf else 8,
     max_features=.75 if video else .5 if rf else .6,n_jobs=4,class_weight='balanced' if video else None,
     random_state=seed+1100 if video else seed,criterion='gini',bootstrap=rf,min_samples_split=2,min_weight_fraction_leaf=0.,
     max_leaf_nodes=None,min_impurity_decrease=0.,ccp_alpha=0.,warm_start=False,oob_score=False,max_samples=None,monotonic_cst=None)
    cls=type('RandomForestClassifier' if rf else 'ExtraTreesClassifier',(),dict(__module__='sklearn.ensemble._forest',get_params=lambda self,deep=False:self.params))
    model=cls();model.params=params;model.n_features_in_=width;model.n_outputs_=1;model.classes_=np.array([0,1]);model.estimators_=[]
    for value in np.random.RandomState(params['random_state']).randint(np.iinfo(np.int32).max,size=n):
        tree=types.SimpleNamespace(max_depth=0,node_count=1,value=np.array([[[.8,.2]]]),children_left=np.array([-1]),children_right=np.array([-1]),feature=np.array([-2]),threshold=np.array([-2.]))
        model.estimators_.append(types.SimpleNamespace(random_state=int(value),tree_=tree))
    return model

class Partitions(unittest.TestCase):
    def test_exact_SHA_alias_excluded_isolation(self):
        r=records();r[5]['sha256']=r[7]['sha256']
        for a,b,c in (([5],[7],[]),([5],[2],[7]),([2],[5],[7])):
            with self.assertRaises(S.A.AuditError):S.I.partition(r,a,b,c)
    def test_23_jobs_and_inner_outer_final_complete(self):
        r=records(23);r[22]['sha256']=r[1]['sha256'];outer=S.A.seeded_folds(r,S.A.content_groups(r,list(range(len(r)))),5,20261002)
        jobs=S.expected_jobs(r,outer);self.assertEqual(len(jobs),23);self.assertEqual(sum(j['scope']==5 for j in jobs),3)
        for j in jobs:S.I.partition(r,j['fit'],j['predict'],j['excluded'])
        for scope in range(5):
            self.assertEqual(sorted(i for j in jobs if j['scope']==scope and j['inner'] is not None for i in j['predict']),sorted(i for i in range(len(r)) if i not in outer[scope]))
        self.assertEqual(sorted(i for j in jobs if j['scope']==5 for i in j['predict']),list(range(len(r))))
    def test_oof_missing_duplicate_extra_rejected(self):
        p=S.A.Probabilities({2:np.zeros(8,np.float32)},{2:.3})
        for parts,ids in (([p,p],[2]),([p],[2,3]),([p],[])):
            with self.assertRaises(S.A.AuditError):S.I.merge_scores(parts,ids)

class PCA(unittest.TestCase):
    def fixture(self):
        r=records();r[2]['sha256']=r[4]['sha256']='a'*64
        for i,data in ((2,[[0,0],[2,0],[10000,10000]]),(4,[[0,0],[2,0],[10000,10000]]),(6,[[8,0],[10,0],[10000,10000]])):
            r[i]['semantic']=np.array(data,np.float32);r[i]['semantic_support']=np.array([1,1,0],bool)
        return r,[2,4,6]
    def test_supported_equal_content_sparse_alias_golden(self):
        r,train=self.fixture();m=S.moments(r,train)
        np.testing.assert_allclose(m['mean'],[5,0]);np.testing.assert_allclose(m['covariance'],[[17,0],[0,0]])
        proof=S.check_pca(pca(2,1,[5.,0.],[[1.,0.]]),m,1);self.assertEqual(proof['content_groups'],2);self.assertEqual(proof['supported_rows'],6)
    def test_PCA_never_reads_heldout_labels_identity_or_unsupported_values(self):
        r,ids=self.fixture();before=S.moments(r,ids)
        class Forbidden(dict):
            def __getitem__(self,k):raise AssertionError('heldout read')
        for i in range(len(r)):
            if i not in ids:r[i]=Forbidden()
        for i in ids:r[i]['semantic'][-1]=[-1e8,1e8];r[i].pop('labels')
        after=S.moments(r,ids);np.testing.assert_array_equal(before['mean'],after['mean']);np.testing.assert_array_equal(before['covariance'],after['covariance'])
    def test_wrong_mean_per_video_weight_or_subspace_rejected(self):
        r,ids=self.fixture();m=S.moments(r,ids)
        for q in (pca(2,1,[11/3,0.],[[1.,0.]]),pca(2,1,[5.,0.],[[0.,1.]]),pca(2,1,[5.,0.],[[2.,0.]])):
            with self.assertRaises(S.A.AuditError):S.check_pca(q,m,1)
    def test_pca_missing_support_nonfinite_schema_order_rejected(self):
        r,ids=self.fixture();r[2]['semantic_support'][:]=False
        with self.assertRaises(S.A.AuditError):S.moments(r,ids)
        r,ids=self.fixture();m=S.moments(r,ids);q=pca(2,1,[5.,0.],[[1.,0.]]);q['semantic_names'].reverse()
        with self.assertRaises(S.A.AuditError):S.check_pca(q,m,1)
        q=pca(2,1,[np.nan,0.],[[1.,0.]])
        with self.assertRaises(S.A.AuditError):S.check_pca(q,m,1)

class Features(unittest.TestCase):
    def fixture(self):
        r=records()[0];ids=np.array([0,2,4,6],np.int64);pairs=ids.reshape(-1,2);v=np.array([[1,10],[3,30]],np.float32);axis=np.arange(8)
        x=np.stack([np.interp(axis,pairs.mean(1),v[:,j]) for j in range(2)],axis=1).astype(np.float32)
        a=dict(signals=x,fps=np.array(10.),semantic_support=axis<=6,semantic_direct=np.isin(axis,ids),sampled_frame_ids=ids,tubelet_frame_ids=pairs,tubelet_vectors=v,source_sha256=np.array(r['sha256']))
        return r,a,dict(sampled_frame_ids=ids.tolist(),tubelet_frame_ids=pairs.tolist())
    def test_interpolation_and_tubelet_mask_goldens(self):
        r,a,e=self.fixture();fields,delta=S.feature_archive(a,e,r,2);self.assertEqual(delta,0.);np.testing.assert_array_equal(fields['semantic'][2],[1.5,15])
    def test_feature_precision_masks_source_interpolation_tamper(self):
        r,a,e=self.fixture()
        bads=[]
        for key,value in (('signals',a['signals'].astype(np.float64)),('source_sha256',np.array('b'*64)),('fps',np.array(11.)),('sampled_frame_ids',np.array([0,2,4,9],np.int64))):
            b=deepcopy(a);b[key]=value;bads.append(b)
        b=deepcopy(a);b['semantic_direct'][1]=True;bads.append(b)
        b=deepcopy(a);b['signals'][2,0]+=.1;bads.append(b)
        for b in bads:
            with self.assertRaises(S.A.AuditError):S.feature_archive(b,e,r,2)
    def test_view_has_no_identity_label_or_unsupported_semantic_influence(self):
        r=records()[0];r['semantic_support'][-2:]=False;r['semantic_direct'][-2:]=False
        old=S.view(r,pca());r['semantic'][-2:]=1e7;r['labels']=~r['labels'];r.update(name='changed',sha256='b'*64,generator='changed')
        new=S.view(r,pca());np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1]);self.assertEqual(old[2:],new[2:])
        np.testing.assert_array_equal(new[0][:,-2:],np.stack([r['semantic_direct'],r['semantic_support']],1))
    def test_feature_order_and_fps_are_explicit(self):
        r=records()[0];r['semantic'][:,0]=np.arange(8);a=S.view(r,pca());r['fps']=20.;b=S.view(r,pca())
        self.assertEqual(a[2][-2:],['sem/direct','sem/support']);self.assertEqual(len(a[3]),len(a[1]));self.assertFalse(np.array_equal(a[0],b[0]))

class Trees(unittest.TestCase):
    def test_independent_threshold_equality_float32_and_float64_scores(self):
        m=dict(kind='forest',n_features=1,trees=[dict(left=[1,-1,-1],right=[2,-1,-1],feature=[0,-2,-2],threshold=[.5,-2.,-2.],value=[.5,.2,.8])])
        p=S.prepare_forest(m,1,1,1);x=np.array([[.5],[.5001]],np.float32)
        np.testing.assert_array_equal(S.forest_scores(p,x),np.array([.2,.8],np.float32));self.assertEqual(S.forest_scores(p,x,False).dtype,np.float64)
        self.assertEqual(S.forest_scores(p,np.empty((0,1),np.float32)).shape,(0,))
    def test_tree_cycles_shared_unreachable_value_index_dtype(self):
        bads=[]
        for k,v in (('left',[0]),('value',[np.nan]),('value',[1.1]),('feature',[-1]),('left',[-1.] )):
            m=forest();m['trees'][0][k]=v;bads.append(m)
        m=forest();m['trees'][0]={k:v+v for k,v in m['trees'][0].items()};bads.append(m)
        for m in bads:
            with self.assertRaises(S.A.AuditError):S.prepare_forest(m,1,1,1)
    def test_tree_count_feature_width_foreign_kind_rejected(self):
        for m in (forest(trees=2),forest(width=2),dict(kind='constant',n_features=1,probability=.2)):
            with self.assertRaises(S.A.AuditError):S.prepare_forest(m,1,1,0)
        p=S.prepare_forest(forest(),1,1,0)
        with self.assertRaises(S.A.AuditError):S.forest_scores(p,np.ones((2,1),np.float64))
    def test_native_original_parameters_seed_and_direct_tree_values(self):
        for alg,video in (('rf',False),('et',False),('rf',True)):
            m=fake_native(alg,video=video);b=S.native_bundle(m,alg,17,1,video);n=128 if video else 160 if alg=='rf' else 192
            np.testing.assert_allclose(S.forest_scores(S.prepare_forest(b,1,n,9),np.ones((2,1),np.float32),False),[.2,.2])
    def test_native_hyperparameter_mutations_rejected(self):
        for key,value in (('random_state',18),('min_samples_leaf',1),('max_features',1.),('n_estimators',159),('bootstrap',False),('class_weight','balanced'),('n_jobs',2)):
            m=fake_native();m.params[key]=value
            with self.assertRaises(S.A.AuditError):S.native_bundle(m,'rf',17,1)
    def test_native_tree_seed_class_mass_and_dimension_mutations(self):
        for mode in ('seed','class','mass','width'):
            m=fake_native()
            if mode=='seed':m.estimators_[0].random_state+=1
            elif mode=='class':m.classes_=np.array([-1,1])
            elif mode=='mass':m.estimators_[0].tree_.value[0,0,1]=np.nan
            else:m.n_features_in_=2
            with self.assertRaises(S.A.AuditError):S.native_bundle(m,'rf',17,1)

class StateSafety(unittest.TestCase):
    def test_hash_first_before_unpickler(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.gz';p.write_bytes(b'wrong')
            with patch.object(S.I.LocalStateUnpickler,'load',side_effect=AssertionError('unpickle executed')) as load:
                with self.assertRaisesRegex(S.A.AuditError,'SHA256'):S.read_states(S.A.Evidence(t),'s.gz','a'*64)
                load.assert_not_called()
    def test_bounded_decompression_before_unpickler(self):
        with tempfile.TemporaryDirectory() as t:
            d=gzip.compress(b'x'*100);(Path(t)/'s.gz').write_bytes(d)
            with patch.object(S,'MAX_BYTES',16),patch.object(S.I.LocalStateUnpickler,'load') as load:
                with self.assertRaisesRegex(S.A.AuditError,'bound'):S.read_states(S.A.Evidence(t),'s.gz',S.A.sha(d))
                load.assert_not_called()
    def test_restricted_globals_and_trailing_bytes(self):
        class Bad:
            def __reduce__(self):return eval,('1+1',)
        for payload in (pickle.dumps({'rf':Bad(),'et':{}}),pickle.dumps({'rf':{},'et':{}})+b'junk'):
            with tempfile.TemporaryDirectory() as t:
                d=gzip.compress(payload);(Path(t)/'s.gz').write_bytes(d)
                with self.assertRaises(S.A.AuditError):S.read_states(S.A.Evidence(t),'s.gz',S.A.sha(d))

class FitTampering(unittest.TestCase):
    def fixture(self,t):
        r=records();r[7]['sha256']=r[5]['sha256'];job=dict(id='s0_inner0',scope=0,inner=0,role='inner_oof',fit=[5,7],predict=[2],excluded=[0,1,3,4,6],seed=20261002)
        pc=pca();states={alg:{'transform':{'pca':pc}} for alg in S.ALGORITHMS}
        prefix=Path(t)/S.RUN/'fits'/job['id'];prefix.parent.mkdir(parents=True)
        prefix.with_suffix('.pkl.gz').write_bytes(gzip.compress(pickle.dumps(states)))
        prefix.with_suffix('.portable.json.gz').write_bytes(gzip.compress(json.dumps({alg:{} for alg in S.ALGORITHMS}).encode()))
        arrays={f'{alg}_frame_2':np.full(8,.2,np.float32) for alg in S.ALGORITHMS};arrays.update({f'{alg}_video_2':np.array(.2,np.float64) for alg in S.ALGORITHMS})
        with prefix.with_suffix('.npz').open('wb') as f:np.savez_compressed(f,**arrays)
        row={**job,'protocol_receipt':S.RECEIPT,'new_member_fits':2,'portable_max_error':3e-9,'seconds':0.,'artifacts':{s:S.A.sha(prefix.with_suffix(s).read_bytes()) for s in ('.npz','.pkl.gz','.portable.json.gz')}}
        for key in ('fit','predict','excluded'):row[key+'_content_sha256']=sorted({r[i]['sha256'] for i in job[key]})
        signed(row);prefix.with_suffix('.json').write_text(json.dumps(row),'utf-8')
        return r,job,prefix,row,arrays
    def check(self,t,r,j):
        prepared=S.prepare_forest(forest(),1,1,0)
        def member(*args):return {'pca':pca(),'frame_feature_names':['f'],'video_feature_names':['v']},{key:(prepared,prepared) for key in ('frame_model','video_model')}
        with patch.object(S,'member_check',side_effect=member),patch.object(S,'view',return_value=(np.zeros((8,1),np.float32),np.zeros(1,np.float32),['f'],['v'])):
            return S.verify_job(S.A.Evidence(t),r,j)
    def test_usable_sparse_fit_alias_fixture_and_both_members_replay(self):
        with tempfile.TemporaryDirectory() as t:
            r,j,p,m,a=self.fixture(t);out,proof=self.check(t,r,j);self.assertEqual(set(out),set(S.ALGORITHMS));self.assertEqual(proof['native_members_replayed'],2);self.assertEqual(proof['PCA'][0]['content_groups'],1)
    def test_resigned_metadata_partition_seed_SHA_protocol_rejected(self):
        for k,v in (('fit',[5]),('seed',1),('excluded',[]),('fit_content_sha256',['b'*64]),('protocol_receipt','b'*64),('new_member_fits',3)):
            with tempfile.TemporaryDirectory() as t:
                r,j,p,m,a=self.fixture(t);m[k]=v;signed(m);p.with_suffix('.json').write_text(json.dumps(m),'utf-8')
                with self.assertRaises(S.A.AuditError):self.check(t,r,j)
    def test_resigned_NPZ_bad_dtype_missing_scalar_nonfinite_or_scores(self):
        for key,value in (('rf_frame_2',np.full(8,.9,np.float32)),('rf_frame_2',np.full(8,np.nan,np.float32)),('et_frame_2',np.full(8,.2,np.float64)),('rf_video_2',np.array([.2])),('et_video_2',np.array(.2,np.float32))):
            with tempfile.TemporaryDirectory() as t:
                r,j,p,m,a=self.fixture(t);a[key]=value
                with p.with_suffix('.npz').open('wb') as f:np.savez_compressed(f,**a)
                m['artifacts']['.npz']=S.A.sha(p.with_suffix('.npz').read_bytes());signed(m);p.with_suffix('.json').write_text(json.dumps(m),'utf-8')
                with self.assertRaises(S.A.AuditError):self.check(t,r,j)
    def test_model_and_portable_hash_drift_rejected_before_read(self):
        for suffix in ('.pkl.gz','.portable.json.gz'):
            with tempfile.TemporaryDirectory() as t:
                r,j,p,m,a=self.fixture(t);p.with_suffix(suffix).write_bytes(b'changed')
                with self.assertRaisesRegex(S.A.AuditError,'SHA256'):self.check(t,r,j)

class Selection(unittest.TestCase):
    def test_two_mode_stable_q20_exact_tie_is_default(self):
        r=records(6);same={i:[(1,2)] if i%2 else [] for i in range(6)}
        a=S.choose_independent(r,list(same),{mode:same for mode in S.MODES},101)
        self.assertEqual(a['mode'],S.MODES[0]);self.assertEqual(a['stable_utility'],1.);self.assertEqual(len(a['configurations']),2)
    def test_strict_heldout_scope_and_no_third_mode(self):
        r=records(4)
        class Forbidden(dict):
            def __getitem__(self,k):raise AssertionError('heldout read')
        r[2]=r[3]=Forbidden();pred={S.MODES[0]:{0:[],1:[]},S.MODES[1]:{0:[],1:[(1,2)]}}
        self.assertEqual(S.choose_independent(r,[0,1],pred,101)['mode'],S.MODES[1])
        for bad in ({**pred,'third':pred[S.MODES[0]]},{**pred,S.MODES[1]:{0:[]}}):
            with self.assertRaises(S.A.AuditError):S.choose_independent(r,[0,1],bad,101)
    def test_utility_golden_original_weights_not_v8_or_frame_bonus(self):
        st=np.array([1,0,0,1,0,0,999,0,0,1,1,0]);self.assertEqual(S.I.utility(st,1,1),.65)
    def test_greedy_matching_order_and_inclusive_union(self):
        y=np.array([1,1,0,1,1,0,0,0],bool);st=S.A.row_statistics([(0,3),(0,1)],y)
        self.assertEqual(tuple(st[:6]),(1,1,1,1,1,1));self.assertEqual(tuple(st[6:9]),(3,1,1))
    def test_every_guard_keeps_frozen_threshold(self):
        m={'iou_0.3':{'f1':S.GUARDS['event_f1_03_min']},'iou_0.5':{'f1':S.GUARDS['event_f1_05_min']},'frame':{'f1':S.GUARDS['frame_f1_min']},'normal':{'false_positive_videos':5},'positive_videos_without_candidate':12}
        self.assertTrue(all(S.guards(m).values()))
        for key,field,value in (('frame','f1',.48),('normal','false_positive_videos',6),('iou_0.5','f1',.34)):
            b=deepcopy(m);b[key][field]=value;self.assertFalse(all(S.guards(b).values()))
        m['positive_videos_without_candidate']=13;self.assertFalse(S.guards(m)['positive_empty_improved_3'])
    def test_fusion_exact_native_float32_frame_order_and_coverage(self):
        p=S.A.Probabilities({0:np.full(8,.2,np.float32)},{0:.2});q=S.A.Probabilities({0:np.full(8,.7,np.float32)},{0:.7});t=S.A.Probabilities({0:np.full(8,.3,np.float32)},{0:.3})
        fused=S.fuse({'rf':p,'et':q},t,[0]);np.testing.assert_array_equal(fused.frame[0],.4*p.frame[0]+.4*q.frame[0]+.2*t.frame[0])
        with self.assertRaises(S.A.AuditError):S.fuse({'rf':p,'et':q},t,[0,1])

class Precision(unittest.TestCase):
    def test_float32_roundtrip_is_measurable_and_not_accepted_as_native64(self):
        raw=.888888888;rounded=float(np.float32(raw));self.assertNotEqual(raw,rounded)
        with self.assertRaises(S.A.AuditError):S.close_probability(rounded,raw,'TCN scalar')
        S.close_probability(raw+1e-15,raw,'TCN scalar')
    def test_repair_probability_tolerance_rejects_smaller_than_general_replay_tolerance(self):
        with self.assertRaises(S.A.AuditError):S.close_probability(.4+5e-10,.4,'repair score')
        for value in (np.nan,np.inf,True):
            with self.assertRaises(S.A.AuditError):S.close_probability(value,.4,'repair score')

class CLI(unittest.TestCase):
    def test_no_RUN_mkdir_or_freeze_reservation(self):
        with tempfile.TemporaryDirectory() as t,patch.object(S,'ROOT',Path(t)),contextlib.redirect_stderr(StringIO()):
            p=Path(t)/S.RUN/'independent_audit.json'
            with patch.object(S,'run_audit') as run:self.assertEqual(S.main(['--output',str(p)]),2);run.assert_not_called()
            self.assertFalse((Path(t)/S.RUN).exists())
    def test_exclusive_existing_report_never_overwrites_or_runs(self):
        with tempfile.TemporaryDirectory() as t,patch.object(S,'ROOT',Path(t)),contextlib.redirect_stderr(StringIO()):
            p=Path(t)/S.RUN/'independent_audit.json';p.parent.mkdir(parents=True);p.write_text('old','utf-8')
            with patch.object(S,'run_audit') as run:self.assertEqual(S.main(['--output',str(p)]),2);run.assert_not_called()
            self.assertEqual(p.read_text('utf-8'),'old')
    def test_missing_input_fails_explicitly_and_snapshot_is_pending(self):
        with patch.object(S,'run_audit',side_effect=S.A.MissingArtifact('not ready')),contextlib.redirect_stdout(StringIO()) as out:
            self.assertEqual(S.main(['--snapshot']),1)
        self.assertEqual(json.loads(out.getvalue())['status'],'failed')
        with patch.object(S,'run_audit',return_value={'status':'pending','deployment_approved':False}),contextlib.redirect_stdout(StringIO()) as out:self.assertEqual(S.main(['--snapshot']),2)
        self.assertEqual(json.loads(out.getvalue())['status'],'pending')
    def test_source_has_no_training_producer_predict_or_choose_calls(self):
        tree=ast.parse(Path(S.__file__).read_text('utf-8'))
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):
                name=n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else ''
                self.assertNotIn(name,{'fit','fit_semantic_member','fit_semantic_pca','predict_semantic_member','choose_mode','select','train_temporal'})

if __name__=='__main__':unittest.main()
