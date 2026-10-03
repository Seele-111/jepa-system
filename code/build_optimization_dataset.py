#!/usr/bin/env python3
"""Freeze a video-held-out development protocol before comparing algorithms."""
import argparse, hashlib, json
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'/'algorithm-opt-2026-10-02'
DEFAULT_CACHE=Path(r'\\wsl.localhost\Ubuntu-24.04\home\zzy\jepa_data\clean79_event_jepa_v2')
DEFAULT_VIDEOS=Path(r'C:\Users\admin\Desktop\测试')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def make_folds(rows,n_folds,seed):
    rng=np.random.default_rng(seed);folds=[[] for _ in range(n_folds)]
    strata={}
    for i,r in enumerate(rows):
        event='normal' if r['event_count']==0 else 'multi' if r['event_count']>1 else 'single'
        strata.setdefault((r['generator'],event),[]).append(i)
    offset=0
    for _,indices in sorted(strata.items()):
        indices=list(rng.permutation(indices))
        for j,i in enumerate(indices):folds[(offset+j)%n_folds].append(int(i))
        offset=(offset+len(indices))%n_folds
    return [sorted(f) for f in folds]


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--videos',type=Path,default=DEFAULT_VIDEOS);p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    card_path=ROOT/'data'/'dataset_cards'/'clean79_diagnostic.json'
    card=json.loads(card_path.read_text(encoding='utf-8'))
    cache_names=json.loads((args.cache/'video_names.json').read_text(encoding='utf-8'))
    jepa=np.load(args.cache/'signals.npz',allow_pickle=False);labels=np.load(args.cache/'labels.npz',allow_pickle=False)
    motion_root=ROOT/'output'/'desktop-test-2026-10-02'
    motion_names=json.loads((motion_root/'raw_signal_video_names.json').read_text(encoding='utf-8'))
    motion=np.load(motion_root/'raw_signal_scores.npz',allow_pickle=False)
    rows=[];arrays={};excluded=[]
    for metadata in card['videos']:
        name=metadata['video_name'];video=args.videos/name
        if not video.is_file():excluded.append({'video_name':name,'reason':'local video missing'});continue
        ci=cache_names.index(name);mi=motion_names.index(name)
        x=jepa[f'arr_{ci}'];y=labels[f'arr_{ci}'];m=motion[f'v{mi:03d}']
        annotation=json.loads(Path(metadata['annotation_file']).read_text(encoding='utf-8'))
        gt=np.zeros(int(annotation['total_frames']),dtype=np.int64)
        for event in annotation.get('annotations',[]):gt[max(0,int(event['start_frame'])):min(len(gt),int(event['end_frame'])+1)]=1
        if not np.array_equal(y,gt):raise ValueError(f'cached label mismatch: {name}')
        if len(x)!=len(y) or len(m)!=len(y) or not np.isfinite(x).all():raise ValueError(f'feature mismatch: {name}')
        i=len(rows)
        arrays[f'jepa_{i}']=x.astype(np.float32);arrays[f'rank_motion_{i}']=m.astype(np.float32);arrays[f'labels_{i}']=gt
        rows.append({'name':name,'input_path':str(video),'sha256':digest(video),'frames':len(gt),'fps':float(annotation['fps']),
                     'generator':metadata['generator'],'prompt_id':metadata['prompt_id'],'event_count':metadata['event_count']})
    # No prompt may cross folds. Known ids are unique here; reject a silent assumption.
    prompt_ids=[r['prompt_id'] for r in rows if r['prompt_id']!='unknown']
    if len(prompt_ids)!=len(set(prompt_ids)):raise ValueError('duplicate prompt groups require grouped splitting')
    folds=make_folds(rows,5,20261002)
    assert sorted(sum(folds,[]))==list(range(len(rows)))
    manifest={'schema_version':'algorithm-opt-dataset-v1','created_at':datetime.now().astimezone().isoformat(timespec='seconds'),
      'role':'development_nested_video_cv_not_blind','protocol':{'outer_folds':5,'inner_folds':3,'seed':20261002,
        'stratification':'generator x normal/single/multi; known prompt ids checked unique',
        'objective':'.5*(F1@IoU.3+F1@IoU.5) - .20*normal_video_false_positive_rate',
        'matching':'same inclusive endpoints and prediction-order greedy as historical project',
        'parameter_selection':'inner held-out videos only; outer labels never select fold parameters',
        'limitations':'dataset previously inspected; development CV is not a fresh external blind test; family exploration may still introduce development selection bias'},
      'rows':rows,'excluded':excluded,'outer_folds':folds,'cache_profile':'legacy true JEPA relative error/event84; not corrected absolute-error extractor',
      'feature_names':{'jepa':json.loads((args.cache/'summary.json').read_text(encoding='utf-8'))['feature_names'],
                       'rank_motion':['frame_diff_rank','flow_rank','mean_rank','max_rank']},
      'source_hashes':{str(card_path):digest(card_path),str(args.cache/'signals.npz'):digest(args.cache/'signals.npz'),
                       str(args.cache/'labels.npz'):digest(args.cache/'labels.npz')},
      'forbidden_feature_sources':['labels','video name/id/hash','generator','prompt id'],
      'training_full215_used':False}
    np.savez_compressed(args.output/'dataset.npz',**arrays)
    (args.output/'dataset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Frozen protocol:',len(rows),'videos',sum(r['frames'] for r in rows),'frames',len(excluded),'missing')
    print('outer sizes:',[len(f) for f in folds],'normal:',[sum(rows[i]['event_count']==0 for i in f) for f in folds])

if __name__=='__main__':main()