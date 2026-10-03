#!/usr/bin/env python3
"""Fixed-protocol modality ablation with identical nested/deployment selectors."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import io, json, multiprocessing as mp, time
import numpy as np
import run_event_compact_experiment_v8 as V8
from optimized_grouped_training import ROOT, NEW, load_grouped, grouped_folds, digest
from optimized_feature_blocks_v10 import RECIPES, fit_block_member
from optimized_training_provenance import collect_new_fit_inputs
from run_compact_experiment_v4 import write_new, object_hash, decode_predictions, configurations
from run_optimization_selection import Probabilities, predictions_json, prediction_summary
from run_algorithm_optimization import _group_metrics
from run_recall_selection import paired_uncertainty

OUT = ROOT / 'output/algorithm-opt-v10-blocks'
RICH = RECIPES[-1]
PARENT_RICH = 'event_compact_rgb_corrected_motion_et'
BASELINE = V8.BASELINE
SOURCES = tuple(dict.fromkeys(('run_feature_blocks_v10.py', 'optimized_feature_blocks_v10.py',
    'optimized_grouped_training.py', 'optimized_training_provenance.py') + V8.SOURCES))
_STATE = None


def protocol():
    return {'schema_version': 'feature-block-ablation-v10',
        'role': 'repeated_development_not_blind_or_confirmatory',
        'hypothesis': 'Test whether verified JEPA/RGB blocks actually help the same compact event-balanced ET pipeline; no larger model or decoder search.',
        'rows': 77, 'content_groups': 76,
        'partition': 'Frozen SHA-grouped outer5 and inner3; aliases and original label conflicts retained.',
        'recipes': list(RECIPES), 'reference_recipe': PARENT_RICH,
        'features': 'Exactly frozen v4 masked compact transform, optionally omit RGB/corrected JEPA; motion includes local camera-compensated signals; PCA16 train-content only.',
        'objective': 'Exactly v8 event_weights, ET192/depth8/leaf16/.5; video ET128/depth4/leaf2/.75, normal mass2.',
        'decoder_configs': list(configurations()),
        'selection': 'Exactly V8.choose top8 pooled/32 content bootstrap q20; same function for each outer inner3 and final full-content inner3. No four-outer inner averages.',
        'seeds': {'inner_split': '20261002+scope*31', 'inner_fit': '20261002+scope*53+k', 'outer_fit': '20261002+scope*53+99', 'selection': '20261002+scope', 'final_scope': 5},
        'nested_new_member_fits': 60, 'reused_rich_member_fits': 20,
        'final_selection_new_member_fits': 12,
        'promotion_guards': {'f1_03_no_worse': 0, 'f1_05_delta': .02, 'frame_delta': -.005, 'normal_fp_delta': 0, 'positive_empty_delta': -3, 'parity_max': 2e-6},
        'limitations': ['Same development videos were repeatedly inspected.', 'Only14 normal annotations.',
            'Offline centered/future/global transforms are not causal streaming.',
            'Historical raw feature provenance limitations unchanged.',
            'Matched selection procedures cannot guarantee equal selected recipes across training sample sizes.',
            'Fixed recipe outer diagnostics cannot be used to override final inner-only selection.']}


def current_evidence():
    source = collect_new_fit_inputs()
    parent = V8.check_receipt()
    files = [V8.OUT / 'training' / f'{fold}_{PARENT_RICH}{ext}' for fold in range(5) for ext in ('.npz', '.json')]
    return {'input_hashes': dict(source.input_hashes), 'inputs_sha256': source.inputs_sha256,
        'input_sources': dict(source.sources), 'sources': {n: digest(ROOT/'code'/n) for n in SOURCES},
        'parent_receipt_sha256': parent['receipt_sha256'],
        'parent_artifacts': {p.relative_to(ROOT).as_posix(): digest(p) for p in files},
        'protocol_sha256': digest(OUT/'protocol.json'), 'baseline_sha256': digest(BASELINE),
        'default_models': {n: digest(ROOT/'models'/n) for n in ('optimized_locator_v1.json', 'optimized_motion_locator_v1.json')}}


def freeze():
    write_new(OUT/'protocol.json', protocol())
    row = current_evidence(); row['receipt_sha256'] = object_hash(row)
    write_new(OUT/'training_receipt.json', row)
    for n in SOURCES:
        target = OUT/'sources'/n; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write((ROOT/'code'/n).read_bytes())
    print('FROZEN', row['receipt_sha256'], flush=True)


def check_receipt():
    row = json.loads((OUT/'training_receipt.json').read_text('utf-8'))
    core = {k:v for k,v in row.items() if k != 'receipt_sha256'}
    if object_hash(core) != row['receipt_sha256'] or core != current_evidence():
        raise ValueError('frozen v10 evidence mismatch')
    if protocol() != json.loads((OUT/'protocol.json').read_text('utf-8')):
        raise ValueError('frozen v10 protocol differs')
    if any(digest(OUT/'sources'/n) != h for n,h in row['sources'].items()):
        raise ValueError('v10 source snapshot differs')
    return row


def partitions(records, ids, scope):
    vals = grouped_folds(records, ids, 3, 20261002 + scope*31)
    return [{'fit': [i for i in ids if i not in set(val)], 'validation': val,
             'seed': 20261002 + scope*53 + k} for k,val in enumerate(vals)]


def select_inner(records, ids, probabilities, scope):
    """Single shared selection procedure; never takes outer predictions/metrics."""
    if set(probabilities) != set(RECIPES): raise ValueError('recipe coverage differs')
    rows = []
    for recipe in RECIPES:
        p = probabilities[recipe]
        if set(p.frame) != set(ids) or set(p.video) != set(ids):
            raise ValueError('inner probability coverage differs')
        row = V8.choose(records, ids, p, 20261002 + scope)
        row['candidate'] = recipe; rows.append(row)
    chosen = max(rows, key=lambda z: (z['stable_utility'], *z['key'][1:], -RECIPES.index(z['candidate'])))
    return chosen, rows


def init_worker():
    global _STATE
    _STATE = (*load_grouped(), check_receipt())


def fit_task(args):
    scope, recipe, final = args
    manifest, records, profiles, receipt = _STATE
    ids = list(range(len(records)))
    val = [] if final else manifest['outer_folds'][scope]
    train = [i for i in ids if i not in set(val)]
    prefix = 'final_selection' if final else 'training'
    path = OUT/prefix/f'{scope}_{recipe}.npz'
    if path.exists() or path.with_suffix('.json').exists(): raise ValueError('refusing existing fit artifact')
    start = time.perf_counter(); arrays = {}; fits = []
    splits = partitions(records, train, scope)
    for k, part in enumerate(splits):
        fp, vp, state = fit_block_member(recipe, records, part['fit'], part['validation'], part['seed'])
        for i in part['validation']:
            arrays[f'inner_frame_{i}'] = fp[i]; arrays[f'inner_video_{i}'] = np.asarray(vp[i])
        fits.append({'partition': k, 'fit_content_sha256': state['fit_content_sha256'],
            'transform_sha256': object_hash(state['transform']), 'frame_features': len(state['transform']['frame_feature_names']),
            'video_features': int(state['video_model'].n_features_in_)})
    if not final:
        fp, vp, state = fit_block_member(recipe, records, train, val, 20261002 + scope*53 + 99)
        for i in val:
            arrays[f'outer_frame_{i}'] = fp[i]; arrays[f'outer_video_{i}'] = np.asarray(vp[i])
        fits.append({'partition': 'outer', 'fit_content_sha256': state['fit_content_sha256'],
            'transform_sha256': object_hash(state['transform']), 'frame_features': len(state['transform']['frame_feature_names']),
            'video_features': int(state['video_model'].n_features_in_)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream: np.savez_compressed(stream, **arrays)
    import sklearn
    row = {'scope': scope, 'recipe': recipe, 'final_selection': final, 'train': train, 'validation': val,
        'inner_partitions': splits, 'fit_evidence': fits, 'outer_seed': None if final else 20261002+scope*53+99,
        'training_receipt_sha256': receipt['receipt_sha256'], 'npz_sha256': digest(path),
        'seconds': time.perf_counter()-start, 'runtime': {'numpy':np.__version__, 'sklearn':sklearn.__version__}}
    row['signature'] = object_hash(row); write_new(path.with_suffix('.json'), row)
    return {'scope': scope, 'recipe': recipe, 'seconds': row['seconds'], 'final_selection':final}


def load_fold(scope, recipe, records, receipt, *, final=False):
    if recipe == RICH and not final:
        return V8.load_new_fold(scope, PARENT_RICH, records, V8.check_receipt())
    path = OUT/('final_selection' if final else 'training')/f'{scope}_{recipe}.npz'
    row = json.loads(path.with_suffix('.json').read_text('utf-8'))
    core = {k:v for k,v in row.items() if k != 'signature'}
    if object_hash(core) != row['signature'] or row['training_receipt_sha256'] != receipt['receipt_sha256']:
        raise ValueError('fit metadata differs')
    manifest = json.loads((NEW/'dataset_manifest.json').read_text('utf-8'))
    val = [] if final else manifest['outer_folds'][scope]
    train = [i for i in range(len(records)) if i not in set(val)]
    expected = partitions(records, train, scope)
    if row['recipe'] != recipe or row['scope'] != scope or row['final_selection'] != final or row['train'] != train or row['validation'] != val or row['inner_partitions'] != expected:
        raise ValueError('cached fit partition differs')
    if row['outer_seed'] != (None if final else 20261002+scope*53+99): raise ValueError('fit seed differs')
    evidence = expected + ([] if final else [{'fit':train, 'validation':val}])
    if len(row['fit_evidence']) != len(evidence): raise ValueError('fit evidence coverage differs')
    for fit, part in zip(row['fit_evidence'], evidence):
        a = {records[i]['sha256'] for i in part['fit']}; b = {records[i]['sha256'] for i in part['validation']}
        if a & b or fit['fit_content_sha256'] != sorted(a): raise ValueError('fit content leakage or provenance differs')
    data = path.read_bytes()
    import hashlib
    if hashlib.sha256(data).hexdigest() != row['npz_sha256']: raise ValueError('fit NPZ differs')
    expected_keys = {f'{kind}_{head}_{i}' for kind, sub in [('inner',train),('outer',val)] for i in sub for head in ['frame','video']}
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != expected_keys: raise ValueError('probability key coverage differs')
        a = {k:archive[k].copy() for k in archive.files}
    for kind, sub in [('inner',train),('outer',val)]:
        for i in sub:
            for head, shape in [('frame',(records[i]['frames'],)),('video',())]:
                x = a[f'{kind}_{head}_{i}']
                if x.shape != shape or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
                    raise ValueError('invalid probability tensor')
    def read(kind, sub):
        return Probabilities({i:a[f'{kind}_frame_{i}'] for i in sub}, {i:float(a[f'{kind}_video_{i}']) for i in sub}, None)
    return read('inner',train), read('outer',val), row


def train_all(workers, *, final=False):
    start = time.perf_counter(); check_receipt(); results = []
    jobs = [(5,r,True) for r in RECIPES] if final else [(f,r,False) for f in range(5) for r in RECIPES if r != RICH]
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init_worker) as pool:
        for future in as_completed([pool.submit(fit_task,j) for j in jobs]):
            result = future.result(); results.append(result); print(json.dumps(result), flush=True)
    check_receipt()
    write_new(OUT/('final_training_report.json' if final else 'training_report.json'),
        {'status':'complete', 'member_fits':12 if final else 60, 'results':results, 'elapsed_seconds':time.perf_counter()-start})


def promotion_guards(metrics, baseline):
    x, y = metrics['all'], baseline['all']
    return {'event_f1_03_no_worse': x['iou_0.3']['f1'] >= y['iou_0.3']['f1'],
        'event_f1_05_improved_02': x['iou_0.5']['f1'] >= y['iou_0.5']['f1'] + .02,
        'frame_f1_drop_at_most_005': x['frame']['f1'] >= y['frame']['f1'] - .005,
        'normal_fp_no_worse': x['normal']['false_positive_videos'] <= y['normal']['false_positive_videos'],
        'positive_empty_improved_3': x['positive_videos_without_candidate'] <= y['positive_videos_without_candidate'] - 3}


def select_all():
    start=time.perf_counter(); receipt=check_receipt(); manifest, records, _=load_grouped()
    selected={}; fixed={r:{} for r in RECIPES}; folds=[]
    for scope,val in enumerate(manifest['outer_folds']):
        train=[i for i in range(len(records)) if i not in set(val)]; inner={}; outer={}; metadata={}
        for recipe in RECIPES:
            inner[recipe],outer[recipe],metadata[recipe]=load_fold(scope,recipe,records,receipt)
        if any(metadata[r]['inner_partitions'] != metadata[RICH]['inner_partitions'] for r in RECIPES):
            raise ValueError('ablation partitions differ')
        chosen, rows=select_inner(records,train,inner,scope)
        pred=decode_predictions(records,val,outer[chosen['candidate']],chosen['config']); selected.update(pred)
        for row in rows:
            fixed[row['candidate']].update(decode_predictions(records,val,outer[row['candidate']],row['config']))
        folds.append({'fold':scope,'train_indices':train,'validation_indices':val,
            'inner_partitions':metadata[RICH]['inner_partitions'],'primary':chosen,'all_inner_selections':rows,
            'primary_predictions':{str(i):pred[i] for i in val}})
        print(json.dumps({'fold':scope,'candidate':chosen['candidate'],'config':chosen['config']}),flush=True)
    baseline=json.loads(BASELINE.read_text('utf-8'))['grouped_v1_baseline']
    bp={p['index']:[tuple(x) for x in p['segments']] for p in baseline['predictions']}
    bm=_group_metrics(records,bp); metrics=_group_metrics(records,selected)
    if bm != baseline['metrics']: raise ValueError('baseline reconstruction differs')
    guards=promotion_guards(metrics,bm)
    diagnostic={r:{'role':'fixed_recipe_outer_diagnostic_not_deployment_selection',
        'metrics':_group_metrics(records,pred),'predictions':predictions_json(records,pred)} for r,pred in fixed.items()}
    row={'schema_version':'feature-block-ablation-v10','status':'complete','role':protocol()['role'],
        'protocol_sha256':digest(OUT/'protocol.json'),'training_receipt_sha256':receipt['receipt_sha256'],
        'folds':folds,'grouped_v1_baseline':baseline,
        'primary_v10_nested':{'metrics':metrics,'predictions':predictions_json(records,selected),'summary':prediction_summary(records,selected)},
        'fixed_candidate_diagnostics_not_for_promotion':diagnostic,'statistical_promotion_checks':guards,
        'statistical_promotion_passed':all(guards.values()),
        'paired_content_bootstrap':paired_uncertainty(records,bp,selected),
        'elapsed_seconds':time.perf_counter()-start}
    check_receipt();write_new(OUT/'report.json',row)
    print(json.dumps({'primary':metrics['all'],'guards':guards,'promotion':all(guards.values())}),flush=True)


def final_select():
    receipt=check_receipt();_,records,_=load_grouped();ids=list(range(len(records)));probabilities={};metadata={}
    for recipe in RECIPES:
        probabilities[recipe],outer,metadata[recipe]=load_fold(5,recipe,records,receipt,final=True)
        if outer.frame or outer.video: raise ValueError('final selection must not consume outer predictions')
    chosen,rows=select_inner(records,ids,probabilities,5)
    nested=json.loads((OUT/'report.json').read_text('utf-8'))
    row={'status':'complete','role':'inner3_only_deployment_choice_not_validation',
        'uses_outer_probabilities':False,'uses_outer_metrics_for_selection':False,
        'selection_function':'run_feature_blocks_v10.select_inner','coverage_per_row':1,
        'inner_partitions':metadata[RICH]['inner_partitions'],'chosen':chosen,'all_inner_selections':rows,
        'outer_choices_diagnostic_only':[x['primary']['candidate'] for x in nested['folds']],
        'same_choice_all_outer':all(x['primary']['candidate']==chosen['candidate'] for x in nested['folds']),
        'training_receipt_sha256':receipt['receipt_sha256']}
    check_receipt();write_new(OUT/'deployment_selection.json',row)
    print(json.dumps({'final_candidate':chosen['candidate'],'config':chosen['config'],'outer_choices':row['outer_choices_diagnostic_only']}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['freeze','train','select','final-train','final-select'],required=True)
    p.add_argument('--workers',type=int,default=3); args=p.parse_args()
    if not 1 <= args.workers <= 4: raise ValueError('bounded workers 1..4')
    if args.phase=='freeze':freeze()
    elif args.phase=='train':train_all(args.workers)
    elif args.phase=='select':select_all()
    elif args.phase=='final-train':train_all(args.workers,final=True)
    else:final_select()

if __name__=='__main__':main()
