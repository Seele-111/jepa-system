"""Same frozen content features; repair only legacy CUDA checkpoint residency.

freeze -> probe (one real video) -> run (validated exclusive resumable writes).
No new weights, predictor, training labels, service restart or proxy fallback.
"""
from __future__ import annotations
import argparse
import inspect
import json
import time
from pathlib import Path
import numpy as np
from optimized_grouped_training import ROOT, load_grouped
from optimized_jepa_extractor import Config, sample_video, isolated_legacy_module
from optimized_baseline_crossfit import digest, object_hash, write_new
from optimized_semantic_encoder_loader import load_encoder_cpu_checkpoint

PARENT=ROOT/'output/baseline-targeted-semantic-readout'
OUT=PARENT/'load-repair'
SOURCES=('extract_baseline_semantic_content_repaired.py','optimized_semantic_encoder_loader.py',
         'optimized_jepa_extractor.py','vjepa_predictor.py')


def local_path(value):
    s=str(value)
    if s.startswith('C:'):
        return Path('/mnt/c')/s[3:].replace('\\','/')
    return Path(value)


def freeze():
    if OUT.exists():
        raise FileExistsError('refuse loader-repair namespace overwrite')
    if len(list((PARENT/'features').glob('*.npz'))) != 0 or (PARENT/'feature_manifest.json').exists():
        raise ValueError('unexpected parent extraction evidence; repair requires zero extracted features')
    manifest, records, _=load_grouped()
    parent=json.loads((PARENT/'extraction_protocol.json').read_text('utf-8'))
    OUT.mkdir(); (OUT/'features').mkdir(); (OUT/'sources').mkdir()
    for name in SOURCES:
        with (OUT/'sources'/name).open('xb') as f:
            f.write((ROOT/'code'/name).read_bytes())
    row={'schema_version':'semantic-content-load-repair-v1',
         'parent_protocol_sha256':digest(PARENT/'extraction_protocol.json'),
         'parent_extractor_sha256':digest(PARENT/'sources/extract_baseline_semantic_content.py'),
         'repair_scope':'CPU mmap checkpoint -> strict BF16 CPU encoder -> CUDA; identical weights/preprocess/feature rule',
         'original_failure':'torch.OutOfMemoryError during legacy torch.load(map_location=DEVICE), before video extraction',
         'parent_features_completed':0,'source_sha256':{str(ROOT/'code'/n):digest(ROOT/'code'/n) for n in SOURCES},
         'dataset_manifest_sha256':digest(ROOT/'output/algorithm-opt-2026-10-02-v2/dataset_manifest.json'),
         'rows':len(records),'content_groups':len({r['sha256'] for r in records}),
         'profile':parent,'role':'development_input_features_not_accuracy',
         'original_outputs_preserved':True,'runtime_defaults_modified':False}
    row['receipt_sha256']=object_hash(row)
    write_new(OUT/'extraction_protocol.json',row)
    print(json.dumps({'status':'frozen','receipt_sha256':row['receipt_sha256']}),flush=True)


def check_protocol():
    row=json.loads((OUT/'extraction_protocol.json').read_text('utf-8'))
    payload={k:v for k,v in row.items() if k!='receipt_sha256'}
    if row['receipt_sha256']!=object_hash(payload):raise ValueError('protocol receipt mismatch')
    for name,sha in row['source_sha256'].items():
        if digest(name)!=sha:raise ValueError('source drift: '+name)
    if digest(PARENT/'extraction_protocol.json')!=row['parent_protocol_sha256']:
        raise ValueError('parent protocol drift')
    if digest(ROOT/'output/algorithm-opt-2026-10-02-v2/dataset_manifest.json')!=row['dataset_manifest_sha256']:
        raise ValueError('dataset manifest drift')
    return row


def validate_cached(i, record, row):
    path=OUT/'features'/f'v{i:03d}.npz'
    metadata=json.loads((OUT/'features'/f'v{i:03d}.json').read_text('utf-8'))
    if metadata['receipt_sha256']!=object_hash({k:v for k,v in metadata.items() if k!='receipt_sha256'}):
        raise ValueError('feature metadata receipt drift')
    if metadata['feature_sha256']!=digest(path) or metadata['source_sha256']!=row['sha256']:
        raise ValueError('feature/source digest drift')
    with np.load(path,allow_pickle=False) as a:
        t=record['frames']; ids=a['sampled_frame_ids']; members=a['tubelet_frame_ids']
        if (a['signals'].shape!=(t,1408) or a['semantic_support'].shape!=(t,) or a['semantic_direct'].shape!=(t,)
            or a['semantic_support'].dtype!=np.bool_ or a['semantic_direct'].dtype!=np.bool_
            or float(a['fps'])!=record['fps'] or str(a['source_sha256'])!=row['sha256']
            or not np.isfinite(a['signals']).all() or not np.isfinite(a['tubelet_vectors']).all()
            or not np.array_equal(ids.reshape(-1,2),members)
            or a['tubelet_vectors'].shape!=(len(members),1408)):
            raise ValueError('cached semantic schema invalid')
    return metadata


def extract(probe=False):
    protocol=check_protocol(); manifest, records, _=load_grouped()
    if (OUT/'feature_manifest.json').exists():raise FileExistsError('complete extraction already frozen')
    if probe and (OUT/'probe_report.json').exists():raise FileExistsError('probe already recorded')
    started=time.perf_counter(); rows=[]; pending=[]; seen={}
    for i,(record,row) in enumerate(zip(records,manifest['rows'])):
        path=OUT/'features'/f'v{i:03d}.npz'; meta=path.with_suffix('.json')
        if path.exists() != meta.exists():raise ValueError('unpaired interrupted feature artifact')
        if path.exists():
            metadata=validate_cached(i,record,row); rows.append(metadata); seen.setdefault(row['sha256'],path)
        else:pending.append(i)
    selected=pending[:1] if probe else pending
    with isolated_legacy_module('vjepa') as legacy:
        import torch
        from src.hub.backbones import vjepa2_1_vit_giant_384
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no proxy fallback')
        checkpoint=Path(protocol['profile']['encoder_checkpoint'])
        if digest(checkpoint)!=protocol['profile']['encoder_checkpoint_sha256']:raise ValueError('checkpoint drift')
        torch.cuda.reset_peak_memory_stats()
        load_start=time.perf_counter()
        scorer=legacy.VJEPASurprise()
        scorer.encoder=load_encoder_cpu_checkpoint(vjepa2_1_vit_giant_384,checkpoint,legacy.DEVICE)
        load_seconds=time.perf_counter()-load_start
        upstream=inspect.getsourcefile(type(scorer.encoder)); upstream_sha=digest(upstream)
        print(json.dumps({'stage':'encoder_loaded','seconds':load_seconds,
                          'cuda_allocated_bytes':int(torch.cuda.memory_allocated())}),flush=True)
        try:
            for i in selected:
                record=records[i]; row=manifest['rows'][i]; source=local_path(row['input_path'])
                if digest(source)!=row['sha256']:raise ValueError('source changed')
                path=OUT/'features'/f'v{i:03d}.npz'; start=time.perf_counter()
                if row['sha256'] in seen:
                    with path.open('xb') as f:f.write(seen[row['sha256']].read_bytes())
                    with np.load(path,allow_pickle=False) as a:
                        ids=a['sampled_frame_ids']; members=a['tubelet_frame_ids']
                    alias=True
                else:
                    sampled=sample_video(source,Config(seed=0,max_frames=32,max_keyframes=8))
                    if sampled.total_frames!=record['frames'] or sampled.fps!=record['fps']:
                        raise ValueError('decoded frame/FPS identity differs')
                    n=len(sampled.frames)//2*2
                    if n<2:raise ValueError('insufficient paired frames')
                    ids=np.asarray(sampled.frame_ids[:n],np.int64); members=ids.reshape(-1,2)
                    video=scorer.preprocess([x[:,:,::-1].copy() for x in sampled.frames[:n]])
                    with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                        tokens=scorer.encoder(video,training=False)
                    if (not isinstance(tokens,torch.Tensor) or tokens.ndim!=3 or tokens.shape[0]!=1
                        or tokens.shape[1]!=len(members)*576 or tokens.shape[2] not in (1408,5632)):
                        raise RuntimeError('encoder content token layout differs')
                    vectors=tokens[0,:,-1408:].reshape(len(members),576,1408).float().mean(1).cpu().numpy()
                    axis=np.arange(sampled.total_frames); centers=members.mean(1)
                    dense=np.stack([np.interp(axis,centers,vectors[:,j]) for j in range(1408)],axis=1).astype(np.float32)
                    support=(axis>=members.min())&(axis<=members.max()); direct=np.isin(axis,ids)
                    if not np.isfinite(dense).all():raise ValueError('nonfinite semantic content')
                    with path.open('xb') as f:
                        np.savez_compressed(f,signals=dense,fps=np.asarray(sampled.fps),
                            semantic_support=support,semantic_direct=direct,sampled_frame_ids=ids,
                            tubelet_frame_ids=members,tubelet_vectors=vectors,source_sha256=np.asarray(row['sha256']))
                    del tokens,video,vectors,dense
                    torch.cuda.empty_cache(); seen[row['sha256']]=path; alias=False
                metadata={'index':i,'source_sha256':row['sha256'],'frames':record['frames'],'fps':record['fps'],
                    'feature_width':1408,'file':str(path.relative_to(OUT)), 'feature_sha256':digest(path),
                    'sampled_frame_ids':ids.tolist(),'tubelet_frame_ids':members.tolist(),
                    'alias_reused':alias,'seconds':time.perf_counter()-start,
                    'protocol_receipt':protocol['receipt_sha256']}
                metadata['receipt_sha256']=object_hash(metadata)
                write_new(path.with_suffix('.json'),metadata)
                rows.append(validate_cached(i,record,row))
                print(json.dumps({'semantic_contents_complete':len(rows),'rows':len(records),'index':i,
                                  'seconds':metadata['seconds']}),flush=True)
        finally:
            scorer._free_encoder()
        if digest(checkpoint)!=protocol['profile']['encoder_checkpoint_sha256']:raise ValueError('checkpoint changed')
        peak=int(torch.cuda.max_memory_allocated())
    if probe:
        write_new(OUT/'probe_report.json',{'status':'passed','videos':selected,'feature_width':1408,
            'finite_and_schema_verified':True,'checkpoint_loaded_strictly':True,'model_dtype':'bfloat16',
            'encoder_upstream_source':upstream,'encoder_upstream_sha256':upstream_sha,
            'cuda_peak_allocated_bytes':peak,'encoder_load_seconds':load_seconds,
            'elapsed_seconds':time.perf_counter()-started,'accuracy_claim':False})
    else:
        rows.sort(key=lambda r:r['index'])
        if [r['index'] for r in rows]!=list(range(len(records))):raise ValueError('incomplete extraction')
        row={'status':'complete','profile':protocol['profile'],'repair_protocol_receipt':protocol['receipt_sha256'],
            'videos':rows,'dataset_manifest_sha256':protocol['dataset_manifest_sha256'],
            'encoder_upstream_source':upstream,'encoder_upstream_sha256':upstream_sha,
            'elapsed_seconds':time.perf_counter()-started,'encoder_load_seconds':load_seconds,
            'cuda_peak_allocated_bytes':peak,'role':'development_input_features_not_accuracy'}
        row['receipt_sha256']=object_hash(row)
        write_new(OUT/'feature_manifest.json',row)
    print(json.dumps({'status':'probe_passed' if probe else 'complete','rows':len(rows),
                      'elapsed_seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=('freeze','probe','run'))
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:extract(args.action=='probe')
