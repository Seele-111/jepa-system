#!/usr/bin/env python3
"""Extract label-free absolute motion features for the frozen development cohort."""
from __future__ import annotations
import argparse, hashlib, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
from optimized_motion_features import extract_video_features, FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''): h.update(part)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=ROOT/'output'/'algorithm-opt-2026-10-02')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    manifest_path = args.data_root/'dataset_manifest.json'
    dataset = json.loads(manifest_path.read_text(encoding='utf-8'))
    out = args.data_root/'motion_features'; out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    profile = {'name': 'absolute-motion-cpu-v1', 'long_edge': 96, 'residual_active_threshold': .05,
               'feature_names': list(FEATURE_NAMES), 'camera_model': 'median_translation',
               'normalization': 'fixed_units_not_per_video', 'fps': 'actual_decoder_verified_against_annotation',
               'code_sha256': digest(Path(__file__).with_name('optimized_motion_features.py'))}
    signature = hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()
    def work(item):
        i, row = item; target = out/f'v{i:03d}.npz'; meta_path = target.with_suffix('.json')
        if target.is_file() and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            if meta.get('profile_signature') == signature and meta.get('source_sha256') == row['sha256']:
                return {**meta, 'cached': True}
        if digest(row['input_path']) != row['sha256']: raise ValueError('source changed: '+row['name'])
        start = time.perf_counter(); features = extract_video_features(row['input_path'])
        x = features['signals']; valid = features['validity']
        if len(x) != row['frames'] or not valid['fps_valid'] or not np.isfinite(x).all():
            raise ValueError('frame/FPS/finite mismatch: '+row['name'])
        if not np.isclose(features['fps'], row['fps'], rtol=1e-4, atol=1e-4):
            raise ValueError('FPS differs from fixed annotation: '+row['name'])
        if features['feature_names'] != list(FEATURE_NAMES): raise ValueError('feature order mismatch')
        np.savez_compressed(target, signals=x, feature_valid=valid['feature_valid'],
                            frame_ids=np.arange(len(x)), frame_times_seconds=features['frame_times_seconds'],
                            fps=np.asarray(features['fps']))
        meta = {'name': row['name'], 'file': target.name, 'status': 'ok', 'frames': len(x),
                'fps': features['fps'], 'source_sha256': row['sha256'], 'profile_signature': signature,
                'feature_sha256': digest(target), 'elapsed_seconds': time.perf_counter()-start,
                'validity': {k: v for k,v in valid.items() if not isinstance(v,np.ndarray)},
                'metadata': features['metadata']}
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'{i+1:02d}/{len(dataset["rows"])} {row["name"]} T={len(x)} {meta["elapsed_seconds"]:.3f}s', flush=True)
        return meta
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        records = list(pool.map(work, enumerate(dataset['rows'])))
    manifest = {'schema_version': 'absolute-motion-feature-cache-v1', 'profile': profile,
                'profile_signature': signature, 'dataset_manifest_sha256': digest(manifest_path),
                'videos': records, 'elapsed_seconds': time.perf_counter()-started,
                'feature_count': len(FEATURE_NAMES), 'frame_count': sum(r['frames'] for r in records),
                'status': 'complete', 'label_free': True}
    (out/'feature_names.json').write_text(json.dumps(list(FEATURE_NAMES),indent=2),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETE', len(records), 'videos', manifest['frame_count'], 'frames', f'{manifest["elapsed_seconds"]:.2f}s')

if __name__ == '__main__': main()
