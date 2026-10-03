"""Spatial JEPA descriptors from the verified raw heatmaps, added to the frozen
compact rich view. The raw heatmaps are label-free; anchor interpolation follows the
existing corrected-feature policy and remains offline/non-causal.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from optimized_compact_features_v4 import content_weights, fit_rgb_pca
from optimized_feature_blocks_v10 import feature_view_blocks
from optimized_locator import rolling_mean, video_features, export_model, portable_predict
from optimized_event_training_v5 import event_weights

SPATIAL_FIELDS = ('patch_support_fraction','positive_mass','p90','p99','centroid_x','centroid_y','spread_x','spread_y','covariance_xy','anisotropy','top10_mass_fraction','top25_mass_fraction','peak_x','peak_y','edge_mass_fraction','center_mass_fraction','entropy_normalized')
ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT/'output/algorithm-opt-2026-10-02/corrected_jepa_features_v2'
KIND = 'spatial-jepa-v13'
RECIPE = 'spatial_jepa_corrected_motion_et'
BASE_RECIPE = 'blocks_corrected_motion_et'


def _finite_heat_stats(heat, valid):
    mask=np.asarray(valid,bool);value=np.asarray(heat,np.float64)
    if value.ndim!=2 or mask.shape!=value.shape or not np.isfinite(value[mask]).all():
        raise ValueError('invalid JEPA heatmap')
    h,w=value.shape; yy,xx=np.mgrid[0:h,0:w]; denom=max(1,h-1); x=xx/max(1,w-1)*2-1; y=yy/denom*2-1
    vals=np.maximum(value[mask],0.0); coords=(x[mask],y[mask]);
    if len(vals)==0:return np.zeros(len(SPATIAL_FIELDS),np.float32)
    total=float(vals.sum());weights=vals/total if total>1e-12 else np.full(len(vals),1/len(vals))
    cx=float(np.dot(weights,coords[0]));cy=float(np.dot(weights,coords[1]))
    dx=coords[0]-cx;dy=coords[1]-cy
    varx=float(np.dot(weights,dx*dx));vary=float(np.dot(weights,dy*dy));cov=float(np.dot(weights,dx*dy))
    trace=varx+vary
    anis=float(np.sqrt(max(0,(varx-vary)**2+4*cov*cov))/max(trace,1e-6))
    order=np.argsort(vals)[::-1];top10=float(vals[order[:max(1,int(np.ceil(.10*len(vals))))]].sum()/max(total,1e-12))
    top25=float(vals[order[:max(1,int(np.ceil(.25*len(vals))))]].sum()/max(total,1e-12))
    p90=float(np.percentile(vals,90));p99=float(np.percentile(vals,99));mx=int(np.argmax(vals));mx_y,mx_x=np.argwhere(mask)[mx]
    edge=((np.abs(coords[0])>.66)|(np.abs(coords[1])>.66));edge_mass=float(weights[edge].sum())
    center_mass=float(weights[(coords[0]**2+coords[1]**2)<=.25].sum())
    entropy=float(-(weights*np.log(np.maximum(weights,1e-12))).sum()/np.log(max(2,len(weights))))
    return np.asarray([float(mask.mean()),total,p90,p99,cx,cy,np.sqrt(max(0,varx)),np.sqrt(max(0,vary)),cov,anis,top10,top25,float(mx_x/max(1,w-1)*2-1),float(mx_y/max(1,h-1)*2-1),edge_mass,center_mass,entropy],np.float32)


def spatial_from_raw(raw_path, frames, fps):
    with np.load(raw_path,allow_pickle=False) as a:
        required=['vjepa_raw_heatmaps','vjepa_patch_valid_mask','vjepa_valid_mask','tubelet_frame_ids','ijepa_raw_heatmaps','ijepa_patch_valid_mask','ijepa_valid_mask','keyframe_ids','frame_ids']
        if any(k not in a.files for k in required):raise ValueError('raw JEPA cache missing spatial fields')
        if not isinstance(frames,int) or frames<1 or not np.isfinite(fps) or fps<=0:raise ValueError('invalid spatial frame/FPS schema')
        if not np.array_equal(a['frame_ids'],np.arange(frames)):raise ValueError('raw frame ids mismatch')
        if 'timestamps_sec' not in a.files or not np.allclose(a['timestamps_sec'],np.arange(frames)/fps,rtol=0,atol=1e-6):raise ValueError('raw timestamp/FPS mismatch')
        groups=[]
        for prefix,centers in [('v',np.asarray(a['tubelet_frame_ids'],np.float64).mean(axis=1)),('i',np.asarray(a['keyframe_ids'],np.float64).reshape(-1))]:
            heat=np.asarray(a[prefix+'jepa_raw_heatmaps'] if prefix=='v' else a['ijepa_raw_heatmaps'])
            valid=np.asarray(a[prefix+'jepa_patch_valid_mask'] if prefix=='v' else a['ijepa_patch_valid_mask'])
            if heat.ndim!=3 or valid.shape!=heat.shape:raise ValueError('raw spatial heatmap dimensions differ')
            observed=np.asarray(a[prefix+'jepa_valid_mask'] if prefix=='v' else a['ijepa_valid_mask'],bool)
            if len(centers)!=len(heat) or len(observed)!=len(heat):raise ValueError('spatial anchor mismatch')
            anchor=np.stack([_finite_heat_stats(h,m) for h,m in zip(heat,valid)],axis=0)
            out=np.zeros((frames,anchor.shape[1]),np.float32);support=np.zeros_like(out,bool)
            good=observed & np.isfinite(centers) & (centers>=0) & (centers<frames)
            if not good.any():raise ValueError('no valid spatial anchors')
            order=np.argsort(centers[good]);loc=centers[good][order];vals=anchor[good][order]
            for j in range(anchor.shape[1]):out[:,j]=np.interp(np.arange(frames),loc,vals[:,j]).astype(np.float32)
            support[(np.arange(frames)>=loc.min())&(np.arange(frames)<=loc.max()),:]=True
            groups.append((out,support,[prefix+'_spatial/'+name for name in SPATIAL_FIELDS]))
        values=np.concatenate([g[0] for g in groups],axis=1);mask=np.concatenate([g[1] for g in groups],axis=1)
    if values.shape[0]!=frames or not np.isfinite(values).all():raise ValueError('invalid spatial features')
    return values.astype(np.float32),mask,[n for g in groups for n in g[2]]


def load_spatial_grouped():
    from optimized_grouped_training import load_grouped
    manifest,records,profiles=load_grouped()
    raw_manifest=__import__('json').loads((RAW_ROOT/'manifest.json').read_text('utf-8'))
    if raw_manifest.get('status')!='complete' or raw_manifest.get('label_free') is not True:raise ValueError('incomplete/non-label-free raw spatial cache')
    entries=raw_manifest['videos']
    if len(entries)!=len(records):raise ValueError('raw spatial row coverage differs')
    for i,r in enumerate(records):
        entry=entries[i]
        if entry['source_sha256']!=r['sha256'] or entry['name']!=r['name'] or entry['frames']!=r['frames'] or entry['fps']!=r['fps']:raise ValueError('raw spatial row identity differs')
        raw=(RAW_ROOT/entry['raw_directory']/'signals.npz').resolve()
        if not raw.is_relative_to(RAW_ROOT.resolve()):raise ValueError('raw spatial path escapes cache')
        if not raw.is_file():raise ValueError('raw spatial cache missing')
        values,mask,names=spatial_from_raw(raw,r['frames'],r['fps'])
        r['spatial']=values;r['spatial_support']=mask;r['spatial_names']=names
    profiles['spatial']={'extractor_sha256':__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),
        'raw_manifest_sha256':__import__('hashlib').sha256((RAW_ROOT/'manifest.json').read_bytes()).hexdigest(),
        'feature_names':records[0]['spatial_names'],'label_free':True}
    return manifest,records,profiles


def _masked_augment(values, support, names, fps):
    x=np.where(support,values,0).astype(np.float32);pair=support & np.vstack([np.zeros((1,x.shape[1]),bool),support[:-1]])
    delta=np.vstack([np.zeros((1,x.shape[1]),np.float32),np.diff(x,axis=0)])*fps;delta=np.where(pair,delta,0)
    z=np.zeros_like(x)
    for j in range(x.shape[1]):
        obs=x[support[:,j],j]
        if len(obs):
            q=np.percentile(obs,[25,75]);scale=max((q[1]-q[0])/1.349,np.std(obs)*.1+1e-5)
            z[support[:,j],j]=np.clip((obs-np.median(obs))/scale,-8,8)
    fs=[x,delta,np.abs(delta),z];out=list(names)+[n+'/'+s for s in ('delta_per_second','abs_delta_per_second','relative_z') for n in names]
    for seconds in [.2,.6]:
        width=max(1,int(round(seconds*fps)));mass=rolling_mean(support.astype(np.float32),width)
        mean=np.divide(rolling_mean(x,width),mass,out=np.zeros_like(x),where=mass>0)
        fs.extend([mean,np.where(support&(mass>0),x-mean,0)])
        out.extend(n+f'/{s}_{seconds}s' for s in ('mean','contrast') for n in names)
    width=max(1,int(round(.3*fps)));padded=np.pad(x,[(width,width),(0,0)],mode='edge').astype(np.float64);pm=np.pad(support.astype(np.float64),[(width,width),(0,0)],mode='edge');cs=np.vstack([np.zeros((1,x.shape[1])),np.cumsum(padded,axis=0)]);ms=np.vstack([np.zeros((1,x.shape[1])),np.cumsum(pm,axis=0)])
    past=(cs[width:width+len(x)]-cs[:len(x)])/width;future=(cs[2*width+1:2*width+1+len(x)]-cs[width+1:width+1+len(x)])/width
    pastm=(ms[width:width+len(x)]-ms[:len(x)])/width;futurem=(ms[2*width+1:2*width+1+len(x)]-ms[width+1:width+1+len(x)])/width
    past=np.divide(past,pastm,out=np.zeros_like(past),where=pastm>0);future=np.divide(future,futurem,out=np.zeros_like(future),where=futurem>0)
    fs.append(np.where((pastm>0)&(futurem>0),future-past,0).astype(np.float32));out.extend(n+'/future_minus_past_0.3s' for n in names)
    fs.append(support.astype(np.float32));out.extend(n+'/support_flag' for n in names)
    result=np.concatenate(fs,axis=1).astype(np.float32)
    return result,out


def feature_view_spatial(recipe,record,pca=None):
    if recipe!=RECIPE:raise ValueError('unknown spatial recipe')
    base,video,names=feature_view_blocks(BASE_RECIPE,record,pca)
    extra,enames=_masked_augment(record['spatial'],record['spatial_support'],record['spatial_names'],float(record['fps']))
    result=np.concatenate([base,extra],axis=1).astype(np.float32)
    video=np.concatenate([video,video_features(record['spatial']),video_features(record['spatial_support'].astype(np.float32))]).astype(np.float32)
    names=names+enames
    if not np.isfinite(result).all() or not np.isfinite(video).all():raise ValueError('nonfinite spatial view')
    return result,video,names


def fit_spatial_member(recipe,records,train,predict,seed,*,full_fit=False):
    if recipe!=RECIPE or not train or not predict:raise ValueError('invalid spatial fit')
    if len(set(train))!=len(train) or len(set(predict))!=len(predict):raise ValueError('duplicate partition')
    if full_fit:
        if set(train)!=set(predict):raise ValueError('full fit mismatch')
    elif {records[i]['sha256'] for i in train}&{records[i]['sha256'] for i in predict}:raise ValueError('content leakage')
    from sklearn.ensemble import ExtraTreesClassifier
    pca=None;ids=sorted(set(train)|set(predict));views={i:feature_view_spatial(recipe,records[i],pca) for i in ids};names=views[ids[0]][2]
    if any(v[2]!=names for v in views.values()):raise ValueError('feature name mismatch')
    x=np.concatenate([views[i][0] for i in train]);y=np.concatenate([records[i]['labels'] for i in train]).astype(np.uint8);weights=event_weights(records,train)
    model=ExtraTreesClassifier(n_estimators=192,max_depth=8,min_samples_leaf=16,max_features=.5,n_jobs=2,random_state=seed);model.fit(x,y,sample_weight=weights)
    video=np.stack([views[i][1] for i in train]);vy=np.asarray([int(np.any(records[i]['labels'])) for i in train]);cw=content_weights(records,train);vw=np.asarray([cw[i]*(2 if vy[j]==0 else 1) for j,i in enumerate(train)])
    vm=ExtraTreesClassifier(n_estimators=128,max_depth=4,min_samples_leaf=2,max_features=.75,class_weight='balanced',n_jobs=2,random_state=seed+1100);vm.fit(video,vy,sample_weight=vw)
    def positive(m,a):
        c=np.flatnonzero(m.classes_==1);return m.predict_proba(a)[:,int(c[0])].astype(np.float32) if len(c) else np.zeros(len(a),np.float32)
    fp={i:positive(model,views[i][0]) for i in predict};vp={i:float(positive(vm,views[i][1][None])[0]) for i in predict}
    state={'recipe':recipe,'transform':{'feature_view':KIND,'pca':pca,'frame_feature_names':names},'frame_model':model,'video_model':vm,'fit_content_sha256':sorted({records[i]['sha256'] for i in train})}
    return fp,vp,state


def export_spatial_member(state):
    return {'recipe':state['recipe'],'transform':state['transform'],'frame_model':export_model(state['frame_model']),'video_model':export_model(state['video_model']),'fit_content_sha256':state['fit_content_sha256']}


def predict_spatial_member(member,record):
    x,v,n=feature_view_spatial(member['recipe'],record,member['transform'].get('pca'))
    if n!=member['transform']['frame_feature_names']:raise ValueError('feature order mismatch')
    return portable_predict(member['frame_model'],x),float(portable_predict(member['video_model'],v[None])[0])
