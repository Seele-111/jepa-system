"""Strict deeper cross-fitting of the unchanged default RF/ET/TCN members.

New model states are saved for replay. Legacy probabilities retain their legacy
provenance role; audit-time hashes are not rewritten as training-time evidence.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import time
import numpy as np
from optimized_grouped_training import ROOT, load_grouped, grouped_folds, fit_member
from optimized_training_provenance import collect_new_fit_inputs
from run_recall_selection import load_fold

OUT = ROOT / 'output/baseline-targeted-interval-quality'
BASE = OUT / 'base_crossfit'
MEMBERS = (('corrected_motion_rf', .4), ('corrected_motion_et', .4), ('rgb_motion_tcn', .2))
SOURCE_NAMES = ('optimized_baseline_crossfit.py', 'optimized_grouped_training.py',
    'run_algorithm_optimization.py', 'run_optimization_selection.py',
    'optimized_feature_view.py', 'optimized_feature_view_v2.py',
    'optimized_locator.py', 'optimized_temporal_head.py', 'optimized_boundary_head.py',
    'optimized_training_provenance.py', 'build_optimization_dataset.py')
STATE = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def specs(records):
    jobs = []
    for scope in range(6):
        if scope < 5:
            # These are the exact legacy member partitions, verified first.
            _, _, meta = load_fold(scope, MEMBERS[0][0], records)
            inners = meta['inner_partitions']
            for recipe, _ in MEMBERS[1:]:
                _, _, other = load_fold(scope, recipe, records)
                if other['inner_partitions'] != inners:
                    raise ValueError('legacy baseline members do not share inner splits')
            excluded_outer = meta['validation']
        else:
            ids = list(range(len(records)))
            vals = grouped_folds(records, ids, 3, 20273000)
            inners = [{'fit': [i for i in ids if i not in set(val)],
                       'validation': val, 'seed': 20273100 + j} for j, val in enumerate(vals)]
            excluded_outer = []
        for inner, part in enumerate(inners):
            fit, val = list(part['fit']), list(part['validation'])
            if scope == 5:
                jobs.append({'id': f'final_inner_{inner}', 'role': 'final_inner_base_validation',
                    'scope': scope, 'inner': inner, 'deep': None, 'fit': fit, 'predict': val,
                    'excluded': [], 'seed': 20273200 + inner})
            deepvals = grouped_folds(records, fit, 3, 20274000 + scope * 10 + inner)
            for deep, dv in enumerate(deepvals):
                jobs.append({'id': f's{scope}_i{inner}_d{deep}',
                    'role': 'deep_oof_base_for_quality_training', 'scope': scope, 'inner': inner,
                    'deep': deep, 'fit': [i for i in fit if i not in set(dv)], 'predict': dv,
                    'excluded': sorted(set(excluded_outer + val)),
                    'seed': 20275000 + scope * 100 + inner * 10 + deep})
    for job in jobs:
        a, b, c = (set(job[k]) for k in ('fit', 'predict', 'excluded'))
        if a & b or (a | b) & c:
            raise ValueError('new deeper partition leaks excluded row')
        ah, bh, ch = ({records[i]['sha256'] for i in s} for s in (a, b, c))
        if ah & bh or (ah | bh) & ch:
            raise ValueError('new deeper partition leaks excluded content')
    if len(jobs) != 57:
        raise ValueError('unexpected baseline cross-fit count')
    return jobs


def freeze():
    if (OUT / 'baseline_protocol.json').exists() or BASE.exists():
        raise FileExistsError('refuse cross-fit freeze overwrite')
    manifest, records, _ = load_grouped()
    evidence = collect_new_fit_inputs()
    jobs = specs(records)
    sources = {str(ROOT / 'code' / name): digest(ROOT / 'code' / name) for name in SOURCE_NAMES}
    protocol = {
        'schema_version': 'targeted-baseline-interval-crossfit-v1',
        'role': 'repeated_development_not_blind',
        'hypothesis': 'Learn candidate interval completeness/IoU on strict OOF scores; freeze default frame members.',
        'rows': len(records), 'content_groups': len({r['sha256'] for r in records}),
        'members': [{'recipe': r, 'weight': w} for r, w in MEMBERS],
        'member_training': 'unchanged v1 fit_member; no event weighting, feature/model/hyperparameter search',
        'legacy_outer5_inner3': manifest['outer_folds'],
        'jobs': jobs, 'new_member_fits': len(jobs) * len(MEMBERS),
        'input_sha256': dict(evidence.input_hashes), 'input_inventory_sha256': evidence.inputs_sha256,
        'source_sha256': sources,
        'default_sha256': {str(ROOT / 'models' / n): digest(ROOT / 'models' / n)
                           for n in ('optimized_locator_v1.json', 'optimized_motion_locator_v1.json')},
        'save_states': 'Every new member state is retained in locally generated pickle/gzip with hash; never load untrusted pickle.',
        'excluded_label_rule': 'Inner head-fit sees only deep OOF generated wholly inside inner-fit. Outer head-fit sees only outer-fit inner OOF.',
        'head_freeze': 'Separate exact module and interval protocol are frozen before first quality fit.',
    }
    protocol['receipt_sha256'] = object_hash(protocol)
    (BASE / 'models').mkdir(parents=True)
    (BASE / 'predictions').mkdir()
    (BASE / 'sources').mkdir()
    for name in SOURCE_NAMES:
        with (BASE / 'sources' / name).open('xb') as f:
            f.write((ROOT / 'code' / name).read_bytes())
    write_new(OUT / 'baseline_protocol.json', protocol)
    print(json.dumps({'status': 'frozen', 'jobs': len(jobs), 'member_fits': protocol['new_member_fits'],
                      'receipt_sha256': protocol['receipt_sha256']}))


def check_protocol(recheck_inputs=False):
    row = json.loads((OUT / 'baseline_protocol.json').read_text('utf-8'))
    signature = row.pop('receipt_sha256')
    if object_hash(row) != signature:
        raise ValueError('cross-fit protocol signature differs')
    row['receipt_sha256'] = signature
    for p, sha in row['source_sha256'].items():
        if digest(Path(p)) != sha:
            raise ValueError('cross-fit source drift: ' + p)
    for p, sha in row['default_sha256'].items():
        if digest(Path(p)) != sha:
            raise ValueError('default model changed')
    if recheck_inputs:
        for p, sha in row['input_sha256'].items():
            q = ROOT / p
            if digest(q) != sha:
                raise ValueError('cross-fit input drift: ' + str(q))
    return row


def init_worker():
    global STATE
    STATE = load_grouped()


def fit_task(job):
    _, records, _ = STATE or load_grouped()
    protocol = check_protocol()
    jpath = BASE / 'predictions' / (job['id'] + '.json')
    npath = jpath.with_suffix('.npz')
    mpath = BASE / 'models' / (job['id'] + '.pkl.gz')
    if any(p.exists() for p in (jpath, npath, mpath)):
        raise FileExistsError('refuse baseline fit cache overwrite: ' + job['id'])
    if job not in protocol['jobs']:
        raise ValueError('unregistered job')
    start = time.perf_counter()
    frames = {}; videos = {}; states = {}
    for recipe, weight in MEMBERS:
        fp, vp, state = fit_member(recipe, records, job['fit'], job['predict'], job['seed'])
        fm, vm, transform, x, vx, extra = state
        states[recipe] = {'frame_model': fm, 'video_model': vm, 'transform': transform,
                         'boundary_models': extra['boundary_models']}
        for i in job['predict']:
            frames[i] = frames.get(i, np.zeros(records[i]['frames'], np.float32)) + weight * fp[i]
            videos[i] = videos.get(i, 0.) + weight * vp[i]
    if set(frames) != set(job['predict']):
        raise ValueError('missing predictions')
    arrays = {}
    for i in job['predict']:
        p = np.asarray(frames[i], np.float32)
        if p.shape != (records[i]['frames'],) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError('invalid base frame probabilities')
        if not np.isfinite(videos[i]) or not 0 <= videos[i] <= 1:
            raise ValueError('invalid video probability')
        arrays[f'frame_{i}'] = p; arrays[f'video_{i}'] = np.asarray(videos[i], np.float64)
    with npath.open('xb') as f:
        np.savez_compressed(f, **arrays)
    with mpath.open('xb') as f:
        with gzip.GzipFile(fileobj=f, mode='wb', compresslevel=1) as z:
            pickle.dump(states, z, protocol=pickle.HIGHEST_PROTOCOL)
    check_protocol()
    metadata = {**job, 'baseline_protocol_sha256': protocol['receipt_sha256'],
        'fit_content_sha256': sorted({records[i]['sha256'] for i in job['fit']}),
        'predict_content_sha256': sorted({records[i]['sha256'] for i in job['predict']}),
        'excluded_content_sha256': sorted({records[i]['sha256'] for i in job['excluded']}),
        'npz_sha256': digest(npath), 'model_state_sha256': digest(mpath),
        'model_state_path': str(mpath.relative_to(ROOT)), 'seconds': time.perf_counter() - start}
    metadata['metadata_sha256'] = object_hash(metadata)
    write_new(jpath, metadata)
    return {'id': job['id'], 'seconds': metadata['seconds'], 'model_state_sha256': metadata['model_state_sha256']}


def train_all(workers):
    protocol = check_protocol(recheck_inputs=True)
    if (OUT / 'base_training_report.json').exists():
        raise FileExistsError('refuse training report overwrite')
    start = time.perf_counter(); results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        futures = [pool.submit(fit_task, job) for job in protocol['jobs']]
        for future in as_completed(futures):
            results.append(future.result())
            print(json.dumps({'base_jobs_complete': len(results), 'total': len(futures)}), flush=True)
    check_protocol(recheck_inputs=True)
    write_new(OUT / 'base_training_report.json', {'status': 'complete', 'new_member_fits': len(results)*3,
        'results': sorted(results, key=lambda r: r['id']), 'elapsed_seconds': time.perf_counter() - start})


def load_job(job_id, records):
    protocol = check_protocol()
    jpath = BASE / 'predictions' / (job_id + '.json')
    row = json.loads(jpath.read_text('utf-8')); signature = row.pop('metadata_sha256')
    if object_hash(row) != signature or row['baseline_protocol_sha256'] != protocol['receipt_sha256']:
        raise ValueError('new base metadata signature differs')
    row['metadata_sha256'] = signature
    job = next(j for j in protocol['jobs'] if j['id'] == job_id)
    for k, value in job.items():
        if row[k] != value:
            raise ValueError('new baseline cache partition/seed differs')
    if digest(jpath.with_suffix('.npz')) != row['npz_sha256'] or digest(ROOT/row['model_state_path']) != row['model_state_sha256']:
        raise ValueError('new base model/cache bytes changed')
    for k in ('fit','predict','excluded'):
        if row[k+'_content_sha256'] != sorted({records[i]['sha256'] for i in row[k]}):
            raise ValueError('new base content mismatch')
    with np.load(jpath.with_suffix('.npz'), allow_pickle=False) as a:
        expected = {f'{kind}_{i}' for kind in ('frame','video') for i in row['predict']}
        if set(a.files) != expected:
            raise ValueError('new base cache OOF keys differ')
        fp = {i: np.asarray(a[f'frame_{i}'],np.float32) for i in row['predict']}
        vp = {i: float(a[f'video_{i}']) for i in row['predict']}
    return fp, vp, row


def main():
    p=argparse.ArgumentParser();p.add_argument('action', choices=('freeze','train'))
    p.add_argument('--workers', type=int, default=2);args=p.parse_args()
    if args.action=='freeze':freeze()
    else:train_all(args.workers)

if __name__=='__main__':main()