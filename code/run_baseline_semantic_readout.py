"""One fixed baseline-targeted representation ablation, never a parameter search.

Two modes only: unchanged deployed recipe/decoder vs identical RF/ET learners
with TRAIN-only PCA16 encoder content appended. Cached RGB TCN is unchanged.
Selection uses inner grouped OOF only; outer labels are for reporting only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import pickle
from pathlib import Path
import time
import numpy as np
import optimized_baseline_crossfit as B
from optimized_grouped_training import ROOT, load_grouped
from optimized_feature_view import feature_view
from optimized_temporal_head import predict_temporal
from optimized_locator import decode, metrics
from optimized_video_statistics import VideoTruth, video_statistics, utility
from run_recall_selection import load_fold, grouped_bootstrap
from run_optimization_selection import Probabilities, blend_probabilities, CANDIDATES, _group_metrics, predictions_json
import optimized_semantic_readout as S

OUT=ROOT/'output/baseline-targeted-semantic-readout'
FEATURES=OUT/'load-repair'
RUN=OUT/'controlled-readout'
FIXED=next(c for c in CANDIDATES if c.name=='corrected_rf_et_rgb_tcn_40_40_20')
MODES=('default_control','semantic_content')
SOURCES=('run_baseline_semantic_readout.py','optimized_semantic_readout.py',
    'optimized_compact_features_v4.py','optimized_feature_view.py','optimized_locator.py',
    'optimized_temporal_head.py','optimized_video_statistics.py','optimized_baseline_crossfit.py',
    'optimized_grouped_training.py','run_recall_selection.py','run_optimization_selection.py',
    'optimized_training_provenance.py','run_algorithm_optimization.py','build_optimization_dataset.py')
STATE=None


def feature_records():
    manifest,records,profiles=load_grouped()
    frozen=json.loads((FEATURES/'feature_manifest.json').read_text('utf-8'))
    if frozen.get('status')!='complete' or frozen['receipt_sha256']!=B.object_hash({k:v for k,v in frozen.items() if k!='receipt_sha256'}):
        raise ValueError('semantic extraction incomplete or manifest receipt drift')
    if len(frozen['videos'])!=len(records):raise ValueError('semantic row count differs')
    for i,(r,entry) in enumerate(zip(records,frozen['videos'])):
        if entry['index']!=i or entry['source_sha256']!=r['sha256']:raise ValueError('semantic source ordering differs')
        rel=Path(entry['file'])
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe semantic path')
        path=FEATURES/rel
        if B.digest(path)!=entry['feature_sha256']:raise ValueError('semantic feature bytes differ')
        with np.load(path,allow_pickle=False) as a:
            x=np.asarray(a['signals'],np.float32); support=a['semantic_support'];direct=a['semantic_direct']
            if (x.shape!=(r['frames'],1408) or not np.isfinite(x).all()
                or support.shape!=(r['frames'],) or direct.shape!=(r['frames'],)
                or support.dtype!=np.bool_ or direct.dtype!=np.bool_ or np.any(direct & ~support)
                or float(a['fps'])!=r['fps'] or str(a['source_sha256'])!=r['sha256']):
                raise ValueError('semantic archive schema differs')
            r.update(semantic=x,semantic_support=support,semantic_direct=direct)
    return manifest,records,profiles,frozen


def isolation(records,fit,predict,excluded):
    sets=[set(x) for x in (fit,predict,excluded)]
    if not sets[0] or not sets[1] or any(a & b for k,a in enumerate(sets) for b in sets[k+1:]):
        raise ValueError('partition index isolation failure')
    content=[{records[i]['sha256'] for i in x} for x in sets]
    if any(a & b for k,a in enumerate(content) for b in content[k+1:]):
        raise ValueError('partition content isolation failure')


def job_specs(records):
    jobs=[]
    for fold in range(5):
        _,_,meta=load_fold(fold,'corrected_motion_rf',records)
        for recipe,_ in B.MEMBERS[1:]:
            _,_,other=load_fold(fold,recipe,records)
            if (meta['inner_partitions']!=other['inner_partitions'] or meta['train']!=other['train']
                or meta['validation']!=other['validation']):raise ValueError('legacy members use different partitions')
        for inner,part in enumerate(meta['inner_partitions']):
            jobs.append({'id':f's{fold}_inner{inner}','scope':fold,'inner':inner,'role':'inner_oof',
                'fit':part['fit'],'predict':part['validation'],'excluded':meta['validation'],
                'seed':20261002+fold*53+inner})
        jobs.append({'id':f's{fold}_outer','scope':fold,'inner':None,'role':'outer_validation',
            'fit':meta['train'],'predict':meta['validation'],'excluded':[],
            'seed':20261002+fold*53+99})
    base=B.check_protocol(recheck_inputs=True)
    for inner in range(3):
        old=next(j for j in base['jobs'] if j['id']==f'final_inner_{inner}')
        jobs.append({'id':f'final_inner_{inner}','scope':5,'inner':inner,'role':'final_deployment_inner_oof',
            'fit':old['fit'],'predict':old['predict'],'excluded':[],'seed':old['seed']})
    if len(jobs)!=23:raise ValueError('expected 23 jobs/46 fixed member fits')
    for job in jobs:isolation(records,job['fit'],job['predict'],job['excluded'])
    return jobs


def freeze():
    if RUN.exists():raise FileExistsError('refuse readout experiment namespace overwrite')
    manifest,records,_,features=feature_records()
    jobs=job_specs(records);default=json.loads((ROOT/'models/optimized_locator_v1.json').read_text('utf-8'))
    if [(m['recipe'],m['weight']) for m in default['members']]!=list(B.MEMBERS):
        raise ValueError('default members changed')
    RUN.mkdir();(RUN/'sources').mkdir();(RUN/'fits').mkdir()
    for name in SOURCES:
        with (RUN/'sources'/name).open('xb') as f:f.write((ROOT/'code'/name).read_bytes())
    prior=ROOT/'output/baseline-targeted-interval-quality/interval-schema-repair/report.json'
    row={'schema_version':'baseline-targeted-semantic-readout-v1','role':'repeated_development_not_blind',
        'hypothesis':'Error statistics lose content needed to distinguish legitimate motion from anomalous transitions; append frozen encoder content without changing base learners.',
        'diagnosis':{'weak_overlap_events':40,'with_event_peak_at_least_0_5':38,
            'weak_events_with_both_end_contrast_above_0_02':8,'events_without_v_direct_sample':0,
            'events_without_i_direct_sample':5,'increase_sampling_not_justified':True},
        'rows':len(records),'content_groups':len({r['sha256'] for r in records}),'jobs':jobs,
        'source_sha256':{str(ROOT/'code'/n):B.digest(ROOT/'code'/n) for n in SOURCES},
        'feature_manifest_sha256':B.digest(FEATURES/'feature_manifest.json'),
        'feature_manifest_receipt':features['receipt_sha256'],
        'base_protocol_receipt':B.check_protocol()['receipt_sha256'],
        'baseline_diagnostic_report_sha256':B.digest(prior),
        'default_model_sha256':{str(ROOT/'models'/n):B.digest(ROOT/'models'/n) for n in ('optimized_locator_v1.json','optimized_motion_locator_v1.json')},
        'default_detector_sha256':B.digest(ROOT/'code/optimized_detector.py'),
        'modes':list(MODES),'members':[{'algorithm':'rf','weight':.4},{'algorithm':'et','weight':.4},{'cached_recipe':'rgb_motion_tcn','weight':.2}],
        'semantic_transform':'supported TRAIN frames only; equal SHA-content covariance PCA16; FPS augment_signals; direct/support flags; original base untouched',
        'learners':'original RF160/depth8/leaf10/.5; ET192/depth9/leaf8/.6; unchanged video ET128/depth4/leaf2/.75/balanced; original per-video positive-balanced frame weights',
        'seeds':'same corresponding baseline per-fit seed; final inner same frozen deeper-crossfit seed',
        'default_decoder':default['decoder'],'decoder_selection':'none; exact existing deployed config',
        'choice':{'order':list(MODES),'bootstrap_replicates':32,'q20':.2,'stable_utility':'.75 pooled + .25 grouped bootstrap q20',
                  'utility':'.35 F1@.3 + .65 F1@.5 - .20 normalFPR - .15 positiveEmptyRate',
                  'tie':'earlier declared mode; no decoder grid, calibration, candidate filter or label-dependent feature'},
        'guards':{'event_f1_03_min':.5034013605442177,'event_f1_05_min':.34653061224489793,
                  'frame_f1_min':.5586221701795472,'normal_fp_max':5,'positive_empty_max':12},
        'evidence_boundary':'77 rows already used for development; exact SHA isolation is not near-duplicate/new-source blind generalization; fresh inference is parity, not accuracy',
        'runtime_defaults_modified':False}
    row['receipt_sha256']=B.object_hash(row);B.write_new(RUN/'protocol.json',row)
    print(json.dumps({'status':'frozen','jobs':23,'new_fixed_member_fits':46,'receipt_sha256':row['receipt_sha256']}),flush=True)


def check_protocol(inputs=False):
    row=json.loads((RUN/'protocol.json').read_text('utf-8'))
    if row['receipt_sha256']!=B.object_hash({k:v for k,v in row.items() if k!='receipt_sha256'}):raise ValueError('readout protocol receipt mismatch')
    for name,sha in row['source_sha256'].items():
        if B.digest(name)!=sha:raise ValueError('readout source drift: '+name)
    for name,sha in row['default_model_sha256'].items():
        if B.digest(name)!=sha:raise ValueError('default model unexpectedly changed')
    if B.digest(ROOT/'code/optimized_detector.py')!=row['default_detector_sha256']:raise ValueError('default detector changed')
    if B.digest(FEATURES/'feature_manifest.json')!=row['feature_manifest_sha256']:raise ValueError('content manifest drift')
    if inputs:
        feature_records();B.check_protocol(recheck_inputs=True)
    return row


def valid_probabilities(fp,vp,ids,records):
    if set(fp)!=set(ids) or set(vp)!=set(ids):raise ValueError('score coverage differs')
    for i in ids:
        x=np.asarray(fp[i])
        if (x.shape!=(records[i]['frames'],) or x.dtype!=np.float32 or not np.isfinite(x).all()
            or np.any((x<0)|(x>1)) or not np.isfinite(vp[i]) or not 0<=vp[i]<=1):
            raise ValueError('invalid heldout probability')


def init_worker():
    global STATE
    check_protocol();STATE=feature_records()


def fit_task(job):
    global STATE
    if STATE is None:init_worker()
    protocol=check_protocol();_,records,_,_=STATE
    if job not in protocol['jobs']:raise ValueError('unregistered readout job')
    isolation(records,job['fit'],job['predict'],job['excluded'])
    prefix=RUN/'fits'/job['id']
    if any(prefix.with_suffix(s).exists() for s in ('.npz','.json','.pkl.gz','.portable.json.gz')):
        raise FileExistsError('refuse fit artifact overwrite')
    started=time.perf_counter();states={};portable={};arrays={};maxerror=0.
    for algorithm in ('rf','et'):
        fp,vp,state=S.fit_semantic_member(algorithm,records,job['fit'],job['predict'],job['seed'])
        valid_probabilities(fp,vp,job['predict'],records);states[algorithm]=state
        portable[algorithm]=S.export_semantic_member(state)
        for i in job['predict']:
            p,v=S.predict_semantic_member(portable[algorithm],records[i])
            maxerror=max(maxerror,float(np.max(np.abs(p-fp[i]))),abs(v-vp[i]))
            if maxerror>2e-6:raise ValueError('portable semantic member parity failure')
            arrays[f'{algorithm}_frame_{i}']=fp[i];arrays[f'{algorithm}_video_{i}']=np.asarray(vp[i],np.float64)
    with prefix.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    with prefix.with_suffix('.pkl.gz').open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',compresslevel=1) as z:pickle.dump(states,z,protocol=pickle.HIGHEST_PROTOCOL)
    with prefix.with_suffix('.portable.json.gz').open('xb') as f:
        with gzip.GzipFile(fileobj=f,mode='wb',compresslevel=1) as z:
            z.write(json.dumps(portable,separators=(',',':'),allow_nan=False).encode())
    row={**job,'protocol_receipt':protocol['receipt_sha256'],
        'fit_content_sha256':sorted({records[i]['sha256'] for i in job['fit']}),
        'predict_content_sha256':sorted({records[i]['sha256'] for i in job['predict']}),
        'excluded_content_sha256':sorted({records[i]['sha256'] for i in job['excluded']}),
        'artifacts':{s:B.digest(prefix.with_suffix(s)) for s in ('.npz','.pkl.gz','.portable.json.gz')},
        'portable_max_error':maxerror,'new_member_fits':2,'seconds':time.perf_counter()-started}
    row['receipt_sha256']=B.object_hash(row);B.write_new(prefix.with_suffix('.json'),row);check_protocol()
    return {'id':job['id'],'seconds':row['seconds'],'portable_max_error':maxerror}


def train_all(workers):
    protocol=check_protocol(inputs=True)
    if (RUN/'training_report.json').exists():raise FileExistsError('training report exists')
    started=time.perf_counter();results=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
        futures=[pool.submit(fit_task,j) for j in protocol['jobs']]
        for task in as_completed(futures):
            results.append(task.result());print(json.dumps({'readout_jobs_complete':len(results),'jobs':len(futures)}),flush=True)
    check_protocol(inputs=True)
    B.write_new(RUN/'training_report.json',{'status':'complete','new_member_fits':len(results)*2,'results':sorted(results,key=lambda r:r['id']),
        'elapsed_seconds':time.perf_counter()-started,'maximum_portable_error':max(r['portable_max_error'] for r in results)})


def load_job(name,records):
    protocol=check_protocol();job=next(j for j in protocol['jobs'] if j['id']==name)
    prefix=RUN/'fits'/name;row=json.loads(prefix.with_suffix('.json').read_text('utf-8'))
    if row['receipt_sha256']!=B.object_hash({k:v for k,v in row.items() if k!='receipt_sha256'}):raise ValueError('fit metadata signature differs')
    if row['protocol_receipt']!=protocol['receipt_sha256'] or any(row[k]!=v for k,v in job.items()):raise ValueError('fit partition/seed drift')
    for s,sha in row['artifacts'].items():
        if B.digest(prefix.with_suffix(s))!=sha:raise ValueError('fit artifact digest drift')
    for k in ('fit','predict','excluded'):
        if row[k+'_content_sha256']!=sorted({records[i]['sha256'] for i in job[k]}):raise ValueError('fit content differs')
    probabilities={}
    with np.load(prefix.with_suffix('.npz'),allow_pickle=False) as a:
        expected={f'{algo}_{kind}_{i}' for algo in ('rf','et') for kind in ('frame','video') for i in job['predict']}
        if set(a.files)!=expected:raise ValueError('readout score archive keys differ')
        for algo in ('rf','et'):
            fp={i:np.asarray(a[f'{algo}_frame_{i}'],np.float32) for i in job['predict']}
            vp={i:float(a[f'{algo}_video_{i}']) for i in job['predict']}
            valid_probabilities(fp,vp,job['predict'],records);probabilities[algo]=Probabilities(fp,vp)
    return probabilities,row


def positive(model,values):
    matches=np.flatnonzero(model.classes_==1)
    if not len(matches):return np.zeros(len(values),np.float32)
    return model.predict_proba(values)[:,int(matches[0])].astype(np.float32)


def final_control_and_tcn(name,records):
    fp,vp,meta=B.load_job(name,records)
    # Locally self-generated states only; B.load_job checks trusted protocol and
    # SHA256 before this replay. Never load third-party pickle.
    path=ROOT/meta['model_state_path']
    if B.digest(path)!=meta['model_state_sha256']:raise ValueError('TCN model-state hash mismatch')
    with gzip.open(path,'rb') as f:state=pickle.load(f)['rgb_motion_tcn']
    frame={};video={};recipe=state['transform']['recipe'];pca=state['transform']['pca']
    for i in meta['predict']:
        x,v,names=feature_view(recipe,records[i],pca)
        if names!=state['transform']['frame_feature_names']:raise ValueError('TCN replay feature order differs')
        frame[i]=predict_temporal(state['frame_model'],x,records[i]['fps'])
        video[i]=float(positive(state['video_model'],v[None])[0])
    valid_probabilities(frame,video,meta['predict'],records)
    return Probabilities(fp,vp),Probabilities(frame,video),meta


def fuse(semantic,tcn,ids,records):
    frame={};video={}
    for i in ids:
        frame[i]=(.4*semantic['rf'].frame[i]+.4*semantic['et'].frame[i]+.2*tcn.frame[i]).astype(np.float32)
        video[i]=.4*semantic['rf'].video[i]+.4*semantic['et'].video[i]+.2*tcn.video[i]
    valid_probabilities(frame,video,ids,records)
    return Probabilities(frame,video)


def decode_scores(records,ids,scores,config):
    return {i:decode(scores.frame[i],records[i]['fps'],scores.video[i],config) for i in ids}


def choose_mode(records,ids,mode_predictions,seed):
    truths=[VideoTruth(records[i]['labels']) for i in ids]
    normal=sum(t.normal for t in truths);positive_count=len(ids)-normal
    boot=grouped_bootstrap(records,ids,seed)
    bn=boot@np.asarray([t.normal for t in truths]);bp=boot@np.asarray([not t.normal for t in truths])
    rows=[]
    for ordinal,mode in enumerate(MODES):
        pred=mode_predictions[mode]
        if set(pred)!=set(ids):raise ValueError('selection missing OOF rows')
        matrix=np.stack([video_statistics(pred[i],t) for i,t in zip(ids,truths)])
        pooled=float(utility(matrix.sum(0),normal,positive_count))
        resampled=boot@matrix
        values=np.asarray([utility(s,n,p) for s,n,p in zip(resampled,bn,bp)])
        q20=float(np.quantile(values,.2));stable=.75*pooled+.25*q20
        rows.append({'mode':mode,'ordinal':ordinal,'pooled_utility':pooled,'bootstrap_q20':q20,'stable_utility':stable,
            'inner_diagnostic_metrics_not_validation':metrics([pred[i] for i in ids],[records[i]['labels'] for i in ids])})
    chosen=max(rows,key=lambda r:(r['stable_utility'],-r['ordinal']))
    return {**chosen,'configurations':rows}


def guards(result,p):
    g=p['guards']
    return {'event_f1_03_no_worse':result['iou_0.3']['f1']>=g['event_f1_03_min'],
            'event_f1_05_improved_02':result['iou_0.5']['f1']>=g['event_f1_05_min'],
            'frame_f1_drop_at_most_005':result['frame']['f1']>=g['frame_f1_min'],
            'normal_fp_no_worse':result['normal']['false_positive_videos']<=g['normal_fp_max'],
            'positive_empty_improved_3':result['positive_videos_without_candidate']<=g['positive_empty_max']}


def select():
    p=check_protocol(inputs=True);_,records,_,_=feature_records()
    if not (RUN/'training_report.json').exists():raise ValueError('readout training report not complete')
    if (RUN/'report.json').exists():raise FileExistsError('refuse selection report overwrite')
    config=p['default_decoder'];outer_predictions={mode:{} for mode in MODES};primary={};folds=[]
    for fold in range(5):
        inner_cache={};outer_cache={};meta=None
        for recipe,_ in B.MEMBERS:
            inside,outside,meta=load_fold(fold,recipe,records);inner_cache[recipe]=inside;outer_cache[recipe]=outside
        train,val=meta['train'],meta['validation']
        disabled_inner=blend_probabilities(FIXED,inner_cache,train)
        disabled_outer=blend_probabilities(FIXED,outer_cache,val)
        newinner={'rf':Probabilities({},{}),'et':Probabilities({},{})}
        for inner in range(3):
            new,row=load_job(f's{fold}_inner{inner}',records)
            for algo in ('rf','et'):
                if set(newinner[algo].frame)&set(new[algo].frame):raise ValueError('duplicate semantic inner validation')
                newinner[algo].frame.update(new[algo].frame);newinner[algo].video.update(new[algo].video)
        enabled_inner=fuse(newinner,inner_cache['rgb_motion_tcn'],train,records)
        inner_predictions={MODES[0]:decode_scores(records,train,disabled_inner,config),
                           MODES[1]:decode_scores(records,train,enabled_inner,config)}
        chosen=choose_mode(records,train,inner_predictions,20273000+fold)
        # Outer labels are read for aggregate reporting only, AFTER the inner
        # choice. No outer score is a decoder/family/deployment selection input.
        outer,_=load_job(f's{fold}_outer',records)
        enabled_outer=fuse(outer,outer_cache['rgb_motion_tcn'],val,records)
        mode_outputs={MODES[0]:decode_scores(records,val,disabled_outer,config),
                      MODES[1]:decode_scores(records,val,enabled_outer,config)}
        for mode in MODES:outer_predictions[mode].update(mode_outputs[mode])
        primary.update(mode_outputs[chosen['mode']])
        folds.append({'fold':fold,'train':train,'validation':val,'chosen':chosen,
            'predictions':{mode:{str(i):mode_outputs[mode][i] for i in val} for mode in MODES}})
    allids=list(range(len(records)))
    if set(primary)!=set(allids):raise ValueError('outer coverage incomplete')
    baseline=json.loads((ROOT/'output/baseline-targeted-interval-quality/interval-schema-repair/report.json').read_text('utf-8'))['historical_grouped_v1_baseline']
    fixed={mode:{'metrics':_group_metrics(records,outer_predictions[mode]),'predictions':predictions_json(records,outer_predictions[mode])} for mode in MODES}
    primary_metrics=_group_metrics(records,primary);g=guards(primary_metrics['all'],p)
    score={'schema_version':'baseline-targeted-semantic-readout-results-v1','status':'complete',
        'role':'repeated_development_not_blind','protocol_receipt':p['receipt_sha256'],'folds':folds,
        'historical_grouped_v1_baseline':baseline,'fixed_default_control':fixed[MODES[0]],
        'fixed_semantic_content_diagnostic_not_selected_deployment':fixed[MODES[1]],
        'primary_nested_representation_selection':{'metrics':primary_metrics,'predictions':predictions_json(records,primary)},
        'statistical_promotion_checks':g,'statistical_promotion_passed':all(g.values()),'default_promoted':False,
        'single_full_fit_model_is_not_OOF':True}
    B.write_new(RUN/'report.json',score)
    final_disabled=Probabilities({},{});final_semantic=Probabilities({},{});partitions=[]
    for inner in range(3):
        name=f'final_inner_{inner}';control,tcn,old=final_control_and_tcn(name,records);new,row=load_job(name,records)
        if row['fit']!=old['fit'] or row['predict']!=old['predict'] or row['seed']!=old['seed']:
            raise ValueError('final readout/control splits or seeds differ')
        enabled=fuse(new,tcn,row['predict'],records)
        for dest,src in ((final_disabled,control),(final_semantic,enabled)):
            if set(dest.frame)&set(src.frame):raise ValueError('duplicate final OOF rows')
            dest.frame.update(src.frame);dest.video.update(src.video)
        partitions.append({'fit':row['fit'],'validation':row['predict'],'seed':row['seed']})
    final_predictions={MODES[0]:decode_scores(records,allids,final_disabled,config),
                       MODES[1]:decode_scores(records,allids,final_semantic,config)}
    chosen=choose_mode(records,allids,final_predictions,20273111)
    decision={'status':'complete','role':'final_inner_OOF_deployment_selection_optimistic_not_validation',
        'protocol_receipt':p['receipt_sha256'],'inner_partitions':partitions,'chosen':chosen,
        'predictions':{mode:{str(i):final_predictions[mode][i] for i in allids} for mode in MODES},
        'statistical_promotion_passed':all(g.values()),'default_promoted':False,'outer_best_not_used':True}
    B.write_new(RUN/'deployment_selection.json',decision)
    if not all(g.values()) or chosen['mode']==MODES[0]:
        B.write_new(RUN/'final_fit_skipped.json',{'status':'skipped','reason':'statistical guards failed' if not all(g.values()) else 'inner OOF selected unchanged baseline',
            'planned_unused_full_member_fits':2,'default_promoted':False,'export_created':False})
    print(json.dumps({'status':'complete','primary':primary_metrics['all'],'fixed_semantic':fixed[MODES[1]]['metrics']['all'],
                      'guards':g,'deployment_choice':chosen['mode'],'default_promoted':False}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','train','select'))
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='train':train_all(args.workers)
    else:select()
