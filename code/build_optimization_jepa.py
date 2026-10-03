#!/usr/bin/env python3
"""Fixed-profile true JEPA batch with one model load per stage; no proxy fallback."""
from __future__ import annotations
import argparse, ast, hashlib, json, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from optimized_jepa_extractor import Config, Evidence, sample_video, run_vjepa, run_ijepa, real_backend, build_artifacts, write_outputs

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()

def patch_descriptors(arrays,kind):
    heat=arrays[kind+'_raw_heatmaps']; valid=arrays[kind+'_patch_valid_mask']
    observed=arrays[kind+'_valid_mask']
    rows=np.zeros((len(heat),5),np.float32)
    for i in np.flatnonzero(observed):
        values=heat[i][valid[i]]
        rows[i]=[float(arrays[kind+'_raw_errors'][i]),float(values.std()),
                 float(np.percentile(values,90)),float(np.percentile(values,99)),float(valid[i].mean())]
    return rows,observed

def corrected_frame_features(arrays):
    """Absolute patch statistics interpolated from observed anchors, with explicit masks.

    Interpolation/edge extension is a label-free modelling assumption, not new JEPA
    observations. A completely missing model is an error, never an all-zero normal.
    """
    total=len(arrays['frame_ids']); axis=np.arange(total)
    columns=[];names=[]
    for kind,prefix in [('vjepa','v'),('ijepa','i')]:
        descriptors,observed=patch_descriptors(arrays,kind)
        if not observed.any():raise ValueError('no observed '+kind+' evidence')
        members=arrays['tubelet_frame_ids'] if kind=='vjepa' else arrays['keyframe_ids'][:,None]
        centers=members.mean(axis=1)
        locations=centers[observed]
        dense=np.stack([np.interp(axis,locations,descriptors[observed,j]) for j in range(5)],axis=1).astype(np.float32)
        direct=np.zeros(total,np.float32);direct[members[observed].reshape(-1)]=1
        supported=((axis>=members[observed].min())&(axis<=members[observed].max())).astype(np.float32)
        columns.extend([dense,direct[:,None],supported[:,None]])
        names.extend([prefix+'_'+suffix for suffix in ['absolute_mean','patch_std','patch_p90','patch_p99','patch_fraction','direct_observation','interpolation_range_supported']])
    result=np.concatenate(columns,axis=1)
    if not np.isfinite(result).all():raise ValueError('invalid corrected feature values')
    return result,names

def dump_part(path,evidence,sampled,signature,row,elapsed,stats):
    arrays={attr: getattr(evidence,attr) for attr in ['raw_errors','relative_scores','valid_mask','raw_heatmaps','patch_valid_mask','patch_counts']}
    for attr in ['direction_raw_errors','direction_valid_mask','direction_counts']:
        value=getattr(evidence,attr)
        if value is not None:arrays[attr]=value
    meta={'signature':signature,'source_sha256':row['sha256'],'kind':evidence.kind,
          'metadata':evidence.metadata,'elapsed_seconds':elapsed,'model_runtime':stats}
    arrays['metadata_json']=np.asarray(json.dumps(meta))
    arrays['sampled_frame_ids']=sampled.frame_ids;arrays['keyframe_ids']=sampled.keyframe_ids
    np.savez_compressed(path,**arrays)

def read_part(path,kind,signature,row):
    with np.load(path,allow_pickle=False) as archive:
        meta=json.loads(str(archive['metadata_json'].item()))
        if meta.get('kind')!=kind:raise ValueError('JEPA stage kind mismatch')
        if meta['signature']!=signature or meta['source_sha256']!=row['sha256']:raise ValueError('stale true JEPA stage: '+row['name'])
        kwargs={attr:archive[attr] for attr in ['raw_errors','relative_scores','valid_mask','raw_heatmaps','patch_valid_mask','patch_counts']}
        kwargs['metadata']=meta['metadata']
        for attr in ['direction_raw_errors','direction_valid_mask','direction_counts']:
            if attr in archive.files:kwargs[attr]=archive[attr]
        return Evidence(kind,**kwargs),meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=ROOT/'output'/'algorithm-opt-2026-10-02')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--max-frames',type=int,default=32)
    p.add_argument('--feature-folder',default='corrected_jepa_features_v2')
    p.add_argument('--reuse-vjepa',type=Path)
    args=p.parse_args(); root=args.data_root/args.feature_folder;root.mkdir(parents=True,exist_ok=True)
    dataset_path=args.data_root/'dataset_manifest.json';dataset=json.loads(dataset_path.read_text(encoding='utf-8'))
    rows=dataset['rows'][:args.limit or None]
    config=Config(seed=0,max_frames=args.max_frames,max_keyframes=8,bidirectional=True,mask_passes=1)
    profile={'name':'corrected-true-jepa-absolute-patch-v1','config':asdict(config),
             'extractor_sha256':digest(Path(__file__).with_name('optimized_jepa_extractor.py')),
             'adapter_sha256':digest(Path(__file__)),
             'missing_policy':'no full-model missing accepted; observed anchor interpolation plus direct/support indicators',
             'loss_units':'V L2 hierarchical error, I normalized-target smoothL1; not directly mixed',
             'legacy_cache_compatible':False,'i_mask_reuse_policy':'explicit_batch_seed_reset_and_restore'}
    signature=hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()
    started=time.perf_counter();paths=[]
    for row in rows:
        path=Path(row['input_path'])
        if str(path).startswith('C:'):path=Path('/mnt/c')/str(path)[3:].replace('\\','/')
        if digest(path)!=row['sha256']:raise ValueError('source changed')
        paths.append(path)
    if args.reuse_vjepa:
        previous=args.reuse_vjepa
        old_manifest=json.loads((previous/'manifest.json').read_text(encoding='utf-8'))
        if old_manifest['profile']['config']!=asdict(config):raise ValueError('cannot reuse V with different sampling/masks/config')
        old_source=previous/'extractor_source_v1.py'
        if digest(old_source)!=old_manifest['profile']['extractor_sha256']:raise ValueError('old extractor snapshot does not match provenance')
        def v_ast(text):
            tree=ast.parse(text)
            tree.body=[node for node in tree.body if getattr(node,'name','') not in ['_RealIJEPA','run_ijepa','seeded_mask_counter']]
            return ast.dump(tree,include_attributes=False)
        if v_ast(old_source.read_text(encoding='utf-8'))!=v_ast(Path(__file__).with_name('optimized_jepa_extractor.py').read_text(encoding='utf-8')):
            raise ValueError('V source changed; reuse is forbidden')
        for i,row in enumerate(rows):
            source=previous/f'v{i:03d}_vjepa.npz';target=root/source.name
            if target.exists():continue
            v,meta=read_part(source,'vjepa',old_manifest['profile_signature'],row)
            with np.load(source,allow_pickle=False) as archive:parts={key:archive[key] for key in archive.files}
            meta.update(signature=signature,reused_vjepa={'file':str(source),'sha256':digest(source),
                        'proof':'entire extractor AST excluding I backend/run_ijepa/seed-counter helper identical',
                        'original_signature':old_manifest['profile_signature']})
            parts['metadata_json']=np.asarray(json.dumps(meta))
            np.savez_compressed(target,**parts)
        print('V stage explicitly reused with AST/source/config/hash proof:',len(rows),flush=True)
    for kind in ['vjepa','ijepa']:
        pending=[i for i in range(len(rows)) if not (root/f'v{i:03d}_{kind}.npz').is_file()]
        if not pending:continue
        with real_backend(kind,config) as backend:
            for i in pending:
                path=paths[i];sampled=sample_video(path,config);begin=time.perf_counter()
                if sampled.total_frames!=rows[i]['frames'] or not np.isclose(sampled.fps,rows[i]['fps']):raise ValueError('source frame/FPS mismatch')
                evidence=(run_vjepa(sampled.frames,sampled.frame_ids,backend,config) if kind=='vjepa' else
                          run_ijepa(sampled.keyframes,sampled.keyframe_ids,backend,config))
                if not evidence.valid_mask.any():raise ValueError('missing '+kind+' '+rows[i]['name'])
                stats={**backend.stats,'cuda_peak_allocated_bytes':int(backend.torch.cuda.max_memory_allocated())}
                elapsed=time.perf_counter()-begin
                dump_part(root/f'v{i:03d}_{kind}.npz',evidence,sampled,signature,rows[i],elapsed,stats)
                print(f'{kind} {i+1:02d}/{len(rows)} {rows[i]["name"]} {elapsed:.2f}s observed={int(evidence.valid_mask.sum())}/{len(evidence.valid_mask)}',flush=True)
    records=[];names=None
    for i,(row,path) in enumerate(zip(rows,paths)):
        sampled=sample_video(path,config)
        v,vm=read_part(root/f'v{i:03d}_vjepa.npz','vjepa',signature,row)
        c,cm=read_part(root/f'v{i:03d}_ijepa.npz','ijepa',signature,row)
        runtime={'models':{'vjepa':vm['model_runtime'],'ijepa':cm['model_runtime']},
                 'timings':{'vjepa_seconds':vm['elapsed_seconds'],'ijepa_seconds':cm['elapsed_seconds']},
                 'batch_models_loaded_once':True}
        arrays,meta=build_artifacts(path,sampled,v,c,config,runtime)
        rawdir=root/f'raw_v{i:03d}'
        if not (rawdir/'signals.npz').is_file():write_outputs(rawdir,arrays,meta)
        x,n=corrected_frame_features(arrays)
        if names is None:names=n
        assert names==n
        target=root/f'v{i:03d}.npz'
        np.savez_compressed(target,signals=x,fps=np.asarray(row['fps']),frame_ids=np.arange(len(x)))
        records.append({'name':row['name'],'file':target.name,'status':'ok','frames':len(x),'fps':row['fps'],
                        'source_sha256':row['sha256'],'feature_sha256':digest(target),'raw_directory':rawdir.name,
                        'coverage':meta['coverage'],'vjepa_seconds':vm['elapsed_seconds'],'ijepa_seconds':cm['elapsed_seconds']})
    profile['feature_names']=names
    manifest={'schema_version':'corrected-true-jepa-feature-cache-v1','profile':profile,'profile_signature':signature,
              'dataset_manifest_sha256':digest(dataset_path),'videos':records,'status':'complete' if len(rows)==len(dataset['rows']) else 'pilot',
              'elapsed_seconds':time.perf_counter()-started,'label_free':True}
    (root/'feature_names.json').write_text(json.dumps(names,indent=2),encoding='utf-8')
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETE',len(records),f'{manifest["elapsed_seconds"]:.2f}s',flush=True)

if __name__=='__main__':main()
