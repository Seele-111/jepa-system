"""Retain frozen V-JEPA last-layer tubelet content, not prediction-error scalars.

Only existing local encoder weights. No predictor, new downloads, train labels,
finetuning, proxy fallback or service restart. Decode/source identity is checked.
"""
from pathlib import Path
import inspect,json,time,hashlib
import numpy as np
from optimized_grouped_training import ROOT,load_grouped
from optimized_jepa_extractor import Config,sample_video,isolated_legacy_module

OUT=ROOT/'output/baseline-targeted-semantic-readout'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1048576),b''):h.update(data)
    return h.hexdigest()


def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def main():
    if OUT.exists():raise FileExistsError('refuse semantic extraction namespace overwrite')
    manifest,records,_=load_grouped();OUT.mkdir();(OUT/'features').mkdir();(OUT/'sources').mkdir()
    for name in ('extract_baseline_semantic_content.py','optimized_jepa_extractor.py','vjepa_predictor.py'):
        with (OUT/'sources'/name).open('xb') as f:f.write((ROOT/'code'/name).read_bytes())
    started=time.perf_counter();rows=[];seen={}
    with isolated_legacy_module('vjepa') as legacy:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('no CUDA; no proxy fallback')
        checkpoint=Path(legacy.ENCODER_CKPT);checkpoint_sha=digest(checkpoint)
        profile={'name':'true-vjepa-final-content-tubelet-mean-v1','model':'existing_vjepa2_1_vit_giant_384',
            'encoder_checkpoint':str(checkpoint),'encoder_checkpoint_sha256':checkpoint_sha,
            'source_sha256':{str(ROOT/'code'/n):digest(ROOT/'code'/n) for n in ('extract_baseline_semantic_content.py','optimized_jepa_extractor.py','vjepa_predictor.py')},
            'preprocessing':'existing encoder RGB ImageNet shortest-side resize/384 center crop',
            'max_frames':32,'feature_rule':'encoder eval, training=False, final1408 channels, spatial mean per actual paired tubelet',
            'interpolation':'actual source-frame tubelet centers, explicit member support/direct flags; endpoint extrapolation is modelled not new observation',
            'labels_used':False,'new_weights_downloaded':False,'device':'cuda','dtype':'bfloat16','reduction_dtype':'float32',
            'role':'development_input_features_not_accuracy'}
        write_new(OUT/'extraction_protocol.json',profile)
        scorer=legacy.VJEPASurprise()
        torch.cuda.reset_peak_memory_stats();scorer._load_encoder()
        upstream=inspect.getsourcefile(type(scorer.encoder))
        profile['encoder_upstream_source']=upstream;profile['encoder_upstream_sha256']=digest(upstream)
        try:
            for i,row in enumerate(manifest['rows']):
                source=Path(row['input_path'])
                if str(source).startswith('C:'):source=Path('/mnt/c')/str(source)[3:].replace('\\','/')
                if digest(source)!=row['sha256']:raise ValueError('source changed')
                path=OUT/'features'/f'v{i:03d}.npz';start=time.perf_counter()
                if row['sha256'] in seen:
                    original=seen[row['sha256']]
                    with path.open('xb') as f:f.write(original.read_bytes())
                    with np.load(path,allow_pickle=False) as a:
                        sampled_ids=a['sampled_frame_ids'].tolist();pairs=a['tubelet_frame_ids'].tolist();dimension=a['signals'].shape[1]
                    alias=True
                else:
                    sampled=sample_video(source,Config(seed=0,max_frames=32,max_keyframes=8))
                    if sampled.total_frames!=records[i]['frames'] or sampled.fps!=records[i]['fps']:
                        raise ValueError('decoded frame/FPS identity differs')
                    n=len(sampled.frames)//2*2
                    if n<2:raise ValueError('insufficient paired frames')
                    ids=np.asarray(sampled.frame_ids[:n],np.int64);members=ids.reshape(-1,2)
                    frames=[x[:,:,::-1].copy() for x in sampled.frames[:n]]
                    video=scorer.preprocess(frames)
                    with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                        tokens=scorer.encoder(video,training=False)
                    if not isinstance(tokens,torch.Tensor) or tokens.ndim!=3 or tokens.shape[0]!=1 or tokens.shape[1]!=len(members)*576 or tokens.shape[2] not in (1408,5632):
                        raise RuntimeError('encoder final content token layout differs')
                    vectors=tokens[0,:,-1408:].reshape(len(members),576,1408).float().mean(1).cpu().numpy()
                    axis=np.arange(sampled.total_frames);centers=members.mean(1)
                    dense=np.stack([np.interp(axis,centers,vectors[:,j]) for j in range(1408)],axis=1).astype(np.float32)
                    support=(axis>=members.min())&(axis<=members.max());direct=np.isin(axis,ids)
                    if not np.isfinite(dense).all():raise ValueError('nonfinite semantic content')
                    with path.open('xb') as f:np.savez_compressed(f,signals=dense,fps=np.asarray(sampled.fps),
                        semantic_support=support,semantic_direct=direct,sampled_frame_ids=ids,tubelet_frame_ids=members,
                        tubelet_vectors=vectors,source_sha256=np.asarray(row['sha256']))
                    dimension=1408;sampled_ids=ids.tolist();pairs=members.tolist();seen[row['sha256']]=path;alias=False
                    del tokens,video,vectors,dense;torch.cuda.empty_cache()
                rows.append({'index':i,'source_sha256':row['sha256'],'frames':records[i]['frames'],'fps':records[i]['fps'],
                    'feature_width':dimension,'file':str(path.relative_to(OUT)),'feature_sha256':digest(path),
                    'sampled_frame_ids':sampled_ids,'tubelet_frame_ids':pairs,'alias_reused':alias,'seconds':time.perf_counter()-start})
                print(json.dumps({'semantic_contents_complete':len(rows),'rows':len(records),'seconds':rows[-1]['seconds']}),flush=True)
        finally:scorer._free_encoder()
        if digest(checkpoint)!=checkpoint_sha:raise ValueError('encoder checkpoint drift')
    write_new(OUT/'feature_manifest.json',{'status':'complete','profile':profile,'videos':rows,
        'dataset_manifest_sha256':digest(ROOT/'output/algorithm-opt-2026-10-02-v2/dataset_manifest.json'),
        'elapsed_seconds':time.perf_counter()-started,'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated())})
    print(json.dumps({'status':'complete','rows':len(rows),'feature_width':1408,'elapsed_seconds':time.perf_counter()-started}))

if __name__=='__main__':main()