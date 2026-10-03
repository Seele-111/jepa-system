#!/usr/bin/env python3
"""Frozen targeted normal-negative cost experiment v9."""
from __future__ import annotations
import argparse, hashlib, io, json, multiprocessing as mp, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import run_compact_experiment_v4 as V4
import run_event_compact_experiment_v8 as PARENT
from optimized_grouped_training import ROOT, NEW, load_grouped, grouped_folds, digest
from optimized_training_provenance import collect_new_fit_inputs
from optimized_normal_cost_v9 import RECIPES, fit_normal_cost_member as fit_compact_member
from run_compact_experiment_v4 import write_new, object_hash, blended, decode_predictions
from run_video_gate_experiment_v6 import utility, BASELINE
from copy import deepcopy
from optimized_compact_features_v4 import content_weights
from optimized_video_statistics import VideoTruth, video_statistics
from run_recall_selection import grouped_bootstrap
from run_recall_selection import paired_uncertainty
from run_optimization_selection import Probabilities, predictions_json, prediction_summary
from run_algorithm_optimization import _group_metrics

OUT=ROOT/'output/algorithm-opt-2026-10-02-v9'
CONTROL='event_compact_rgb_corrected_motion_et'
CANDIDATES=tuple((r,((r,1.),)) for r in RECIPES)+(('event_compact_normal2_control',((CONTROL,1.),)),)
SOURCES=('run_normal_cost_experiment_v9.py','optimized_normal_cost_v9.py')+PARENT.SOURCES
_STATE=None
configurations=V4.configurations


def choose(records,ids,p,seed):
    cw=content_weights(records,ids);weights=np.array([cw[i] for i in ids]);truths=[VideoTruth(records[i]['labels']) for i in ids]
    normals=np.array([t.normal for t in truths],float);positives=1-normals
    nm,pm=weights@normals,weights@positives;boot=grouped_bootstrap(records,ids,seed,32)*weights[None,:];bn,bp=boot@normals,boot@positives
    short=[];seen=set()
    for ordinal,cfg in enumerate(configurations()):
        pred=decode_predictions(records,ids,p,cfg);fingerprint=json.dumps([pred[i] for i in ids])
        if fingerprint in seen:continue
        seen.add(fingerprint);mat=np.stack([video_statistics(pred[i],t) for i,t in zip(ids,truths)]);total=weights@mat;u=float(utility(total,nm,pm));f05=2*total[3]/max(1e-12,2*total[3]+total[4]+total[5])
        row={'config':cfg,'stats':total.tolist(),'pooled_utility':u,'key':[u,float(f05),-total[10],-total[9],-total[11],-ordinal],'matrix':mat}
        short.append(row);short.sort(key=lambda z:z['key'],reverse=True);del short[8:]
    for row in short:
        row['bootstrap_q20']=float(np.quantile(utility(boot@row['matrix'],bn,bp),.2));row['stable_utility']=.75*row['pooled_utility']+.25*row['bootstrap_q20'];del row['matrix']
    result=deepcopy(max(short,key=lambda z:(z['stable_utility'],*z['key'][1:])));result['shortlist']=short;return result




def protocol():
    return {'schema_version':'normal-negative-cost-v9','date':'2026-10-02',
            'role':'iterative_development_validation_not_new_blind_test',
            'rationale':'v8 crossed representation/objective reduces empty positives to12 but normal FP remains6. Test one fixed targeted change: normal-content frame negative mass2 to4; do not raise abnormal background costs or tune error-specific features. This is adaptive development after v8, not independent statistical confirmation.',
            'data':'same77 rows/76 SHA, unchanged outer5/inner3 indices,seeds,conflicting aliases retained',
            'frame_weights':'v5 event masses with normal negative4 instead of2; event positive1 split equally, abnormal background1, equal content/alias sharing, class balancing and mean1. Implemented by doubling only normal rows in v5 weights then rebalancing globally; equivalent to raw normal4.',
            'normal_cost_candidates':[2,4], 'new_recipe':list(RECIPES),
            'features_and_models':'Identical v8 compact RGB+JEPA+masked local ET192/depth8/minleaf16/.5,train-content-only PCA16; no other hyperparameter change. Video head remains normal2 ET128/depth4/leaf2/.75,class-balanced.',
            'primary_candidates':[[n,[list(x) for x in m]] for n,m in CANDIDATES],
            'decoder_configs':list(V4.configurations()),
            'selection':'Same v8 content-weighted inner utility/top8/32bootstrap q20 and seeds; no outer-informed candidate/threshold choice',
            'promotion_guards':PARENT.protocol()['promotion_guards'],'training_jobs':20,
            'deployment_selection':'Only four-inner-OOF averages; default promotion remains separately guarded',
            'limitations':PARENT.protocol()['limitations']}


def current_evidence():
    source=collect_new_fit_inputs(); parent=PARENT.check_receipt()
    files=[PARENT.OUT/'training'/f'{fold}_{CONTROL}{ext}' for fold in range(5) for ext in ['.npz','.json']]
    return {'input_hashes':dict(source.input_hashes),'inputs_sha256':source.inputs_sha256,
            'input_source_hashes':dict(source.sources),'sources':{n:digest(ROOT/'code'/n) for n in SOURCES},
            'control_receipt_sha256':parent['receipt_sha256'],'control_artifacts':{p.relative_to(ROOT).as_posix():digest(p) for p in files},
            'protocol_sha256':digest(OUT/'protocol.json'),'baseline_report_sha256':digest(BASELINE)}


def freeze():
    write_new(OUT/'protocol.json',protocol());row=current_evidence();row['receipt_sha256']=object_hash(row)
    write_new(OUT/'training_receipt.json',row)
    for n in SOURCES:
        target=OUT/'sources'/n;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write((ROOT/'code'/n).read_bytes())
    print('FROZEN',row['receipt_sha256'],flush=True)


def check_receipt():
    row=json.loads((OUT/'training_receipt.json').read_text('utf-8'));core={k:v for k,v in row.items() if k!='receipt_sha256'}
    if object_hash(core)!=row['receipt_sha256'] or core!=current_evidence() or protocol()!=json.loads((OUT/'protocol.json').read_text('utf-8')):raise ValueError('v9 frozen receipt mismatch')
    if any(digest(OUT/'sources'/n)!=h for n,h in row['sources'].items()):raise ValueError('v9 source snapshot mismatch')
    return row


def init_worker():
    global _STATE
    _STATE=(*load_grouped(),check_receipt())


def fit_task(args):
    fold, recipe = args
    manifest, records, profiles, receipt = _STATE
    file = OUT / 'training' / f'{fold}_{recipe}.npz'
    meta = file.with_suffix('.json')
    if file.exists() or meta.exists():
        raise ValueError('existing training artifact; never overwrite/refit in a frozen namespace')
    start = time.perf_counter()
    val = manifest['outer_folds'][fold]
    train = [i for i in range(len(records)) if i not in set(val)]
    splits = grouped_folds(records, train, 3, 20261002 + fold * 31)
    arrays, partitions, fits = {}, [], []
    for k, inner_val in enumerate(splits):
        inner_train = [i for i in train if i not in set(inner_val)]
        seed = 20261002 + fold * 53 + k
        fp, vp, state = fit_compact_member(recipe, records, inner_train, inner_val, seed)
        for i in inner_val:
            arrays[f'inner_frame_{i}'] = fp[i]; arrays[f'inner_video_{i}'] = np.asarray(vp[i])
        partitions.append({'fit': inner_train, 'validation': inner_val, 'seed': seed})
        fits.append({'partition': k, 'fit_content_sha256': state['fit_content_sha256'],
                     'transform_sha256': object_hash(state['transform'])})
    seed = 20261002 + fold * 53 + 99
    fp, vp, state = fit_compact_member(recipe, records, train, val, seed)
    for i in val:
        arrays[f'outer_frame_{i}'] = fp[i]; arrays[f'outer_video_{i}'] = np.asarray(vp[i])
    fits.append({'partition': 'outer', 'fit_content_sha256': state['fit_content_sha256'],
                 'transform_sha256': object_hash(state['transform'])})
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    row = {'fold': fold, 'recipe': recipe, 'train': train, 'validation': val,
           'inner_partitions': partitions, 'outer_seed': seed, 'fit_evidence': fits,
           'training_receipt_sha256': receipt['receipt_sha256'], 'npz_sha256': digest(file),
           'seconds': time.perf_counter() - start, 'runtime': {'numpy': np.__version__}}
    import sklearn
    row['runtime']['sklearn'] = sklearn.__version__
    row['signature'] = object_hash(row)
    write_new(meta, row)
    return {'fold': fold, 'recipe': recipe, 'seconds': row['seconds'], 'npz_sha256': row['npz_sha256']}


def load_new_fold(fold, recipe, records, receipt):
    if recipe not in RECIPES:
        raise ValueError('unknown cached recipe')
    file = OUT / 'training' / f'{fold}_{recipe}.npz'
    row = json.loads(file.with_suffix('.json').read_text('utf-8'))
    core = {k: v for k, v in row.items() if k != 'signature'}
    if object_hash(core) != row['signature'] or row['training_receipt_sha256'] != receipt['receipt_sha256']:
        raise ValueError('fit metadata signature mismatch')
    data = file.read_bytes()
    if hashlib.sha256(data).hexdigest() != row['npz_sha256']:
        raise ValueError('fit NPZ mismatch')
    import io
    manifest = json.loads((NEW / 'dataset_manifest.json').read_text('utf-8'))
    val = manifest['outer_folds'][fold]; train = [i for i in range(len(records)) if i not in set(val)]
    if row['recipe'] != recipe or row['fold'] != fold or row['train'] != train or row['validation'] != val or row['outer_seed'] != 20261002 + fold * 53 + 99:
        raise ValueError('fit outer partition mismatch')
    splits = grouped_folds(records, train, 3, 20261002 + fold * 31)
    expected = [{'fit': [i for i in train if i not in set(v)], 'validation': v,
                 'seed': 20261002 + fold * 53 + k} for k, v in enumerate(splits)]
    if row['inner_partitions'] != expected:
        raise ValueError('fit inner partition mismatch')
    for k, part in enumerate(expected + [{'fit': train, 'validation': val}]):
        if {records[i]['sha256'] for i in part['fit']} & {records[i]['sha256'] for i in part['validation']}:
            raise ValueError('cache content leakage')
        fitids = sorted({records[i]['sha256'] for i in part['fit']})
        if row['fit_evidence'][k]['fit_content_sha256'] != fitids:
            raise ValueError('cache PCA/fit training content mismatch')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        expected_keys = {f'{kind}_{head}_{i}' for kind, ids in [('inner', train), ('outer', val)] for i in ids for head in ['frame', 'video']}
        if set(archive.files) != expected_keys:
            raise ValueError('cache probability coverage mismatch')
        a = {k: archive[k].copy() for k in archive.files}
    for kind, ids in [('inner', train), ('outer', val)]:
        for i in ids:
            for head, shape in [('frame', (records[i]['frames'],)), ('video', ())]:
                x = a[f'{kind}_{head}_{i}']
                if x.shape != shape or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
                    raise ValueError('invalid cached probabilities')
    def read(kind, ids):
        return Probabilities({i: a[f'{kind}_frame_{i}'] for i in ids},
                             {i: float(a[f'{kind}_video_{i}']) for i in ids}, None)
    return read('inner', train), read('outer', val), row


def train_all(workers):
    started = time.perf_counter(); check_receipt()
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init_worker) as pool:
        futures = [pool.submit(fit_task, (f, r)) for f in range(5) for r in RECIPES]
        for future in as_completed(futures):
            row = future.result(); results.append(row); print(json.dumps(row), flush=True)
    check_receipt()
    write_new(OUT / 'training_report.json', {'status': 'complete', 'results': results,
              'before_after_evidence_equal': True, 'elapsed_seconds': time.perf_counter() - started})



def select_all():
    start=time.perf_counter();receipt=check_receipt();manifest,records,_=load_grouped();selected={};folds=[];fixed={n:{} for n,m in CANDIDATES}
    for fold,val in enumerate(manifest['outer_folds']):
        train=[i for i in range(len(records)) if i not in set(val)];inner={};outer={};metadata={}
        for r in RECIPES:inner[r],outer[r],metadata[r]=load_new_fold(fold,r,records,receipt)
        inner[CONTROL],outer[CONTROL],control=PARENT.load_new_fold(fold,CONTROL,records,PARENT.check_receipt())
        for m in metadata.values():
            if any(m[k]!=control[k] for k in ['train','validation','inner_partitions']):raise ValueError('v9/control partitions mismatch')
            if [x['fit_content_sha256'] for x in m['fit_evidence']]!=[x['fit_content_sha256'] for x in control['fit_evidence']]:raise ValueError('v9/control fit content mismatch')
        rows=[]
        for n,members in CANDIDATES:
            d=choose(records,train,blended(members,inner,train),20261002+fold);d['candidate']=n;rows.append(d)
        chosen=max(rows,key=lambda z:(z['stable_utility'],*z['key'][1:],-[n for n,m in CANDIDATES].index(z['candidate'])))
        pred=decode_predictions(records,val,blended(dict(CANDIDATES)[chosen['candidate']],outer,val),chosen['config']);selected.update(pred)
        for d in rows:fixed[d['candidate']].update(decode_predictions(records,val,blended(dict(CANDIDATES)[d['candidate']],outer,val),d['config']))
        folds.append({'fold':fold,'train_indices':train,'validation_indices':val,'inner_partitions':control['inner_partitions'],'primary':chosen,'all_inner_selections':rows,'primary_predictions':{str(i):pred[i] for i in val}})
        print(json.dumps({'fold':fold,'candidate':chosen['candidate'],'config':chosen['config']}),flush=True)
    baseline=json.loads(BASELINE.read_text('utf-8'));bp={p['index']:[tuple(x) for x in p['segments']] for p in baseline['grouped_v1_baseline']['predictions']}
    bm=_group_metrics(records,bp);m=_group_metrics(records,selected);x,y=m['all'],bm['all']
    if bm!=baseline['grouped_v1_baseline']['metrics']:raise ValueError('baseline reconstruction mismatch')
    guards={'event_f1_03_no_worse':x['iou_0.3']['f1']>=y['iou_0.3']['f1'],'event_f1_05_improved_02':x['iou_0.5']['f1']>=y['iou_0.5']['f1']+.02,
            'frame_f1_drop_at_most_005':x['frame']['f1']>=y['frame']['f1']-.005,'normal_fp_no_worse':x['normal']['false_positive_videos']<=y['normal']['false_positive_videos'],
            'positive_empty_improved_3':x['positive_videos_without_candidate']<=y['positive_videos_without_candidate']-3}
    report={'schema_version':'normal-negative-cost-v9','status':'complete','role':'iterative_development_validation_not_new_blind_test',
            'protocol_sha256':digest(OUT/'protocol.json'),'training_receipt_sha256':receipt['receipt_sha256'],'folds':folds,
            'grouped_v1_baseline':{'metrics':bm,'predictions':predictions_json(records,bp)},
            'primary_v9_nested':{'metrics':m,'predictions':predictions_json(records,selected),'summary':prediction_summary(records,selected)},
            'fixed_candidate_diagnostics_not_for_promotion':{c:{'metrics':_group_metrics(records,p),'predictions':predictions_json(records,p)} for c,p in fixed.items()},
            'statistical_promotion_checks':guards,'statistical_promotion_passed':all(guards.values()),
            'paired_content_bootstrap':paired_uncertainty(records,bp,selected),'elapsed_seconds':time.perf_counter()-start}
    check_receipt();write_new(OUT/'report.json',report);print(json.dumps({'primary':x,'guards':guards}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['freeze','train','select'],required=True);p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    if args.phase=='freeze':freeze()
    elif args.phase=='train':train_all(args.workers)
    else:select_all()

if __name__=='__main__':main()
