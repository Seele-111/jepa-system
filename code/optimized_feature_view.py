"""Shared label-free training/inference feature view, including fold-fitted PCA."""
from __future__ import annotations
import numpy as np
from optimized_locator import augment_signals, video_features


def temporal_features(values, names, fps):
    values=np.asarray(values,np.float32)
    # No global identity feature is injected into the temporal head. The independent
    # video rejection head gets label-free global statistics through video_features.
    delta=np.vstack([np.zeros((1,values.shape[1]),np.float32),np.diff(values,axis=0)])*fps
    median=np.median(values,axis=0);q=np.percentile(values,[25,75],axis=0)
    scale=np.maximum((q[1]-q[0])/1.349,np.std(values,axis=0)*.1+1e-5)
    z=np.clip((values-median)/scale,-8,8)
    return np.concatenate([values,delta,z],axis=1).astype(np.float32), list(names)+[n+'/delta_per_second' for n in names]+[n+'/relative_robust_z' for n in names]


def feature_view(recipe, record, pca=None):
    fps=float(record['fps']);fs=[];vs=[];names=[]
    if not np.isfinite(fps) or fps<=0:raise ValueError('invalid FPS')
    temporal=recipe.endswith('_tcn')
    use_corrected='corrected' in recipe
    use_jepa=recipe.startswith('jepa') or ('hybrid' in recipe and not use_corrected)
    use_motion=('motion' in recipe or 'hybrid' in recipe) and not recipe.startswith('motion_rank')
    use_rgb=recipe.startswith('rgb_')
    def append(values, raw_names):
        values=np.asarray(values,np.float32)
        if values.ndim!=2 or not len(values) or values.shape[1]<1 or len(raw_names)!=values.shape[1] or not np.isfinite(values).all():
            raise ValueError('invalid input feature schema')
        val,n=(temporal_features(values,raw_names,fps) if temporal else augment_signals(values,raw_names,fps))
        fs.append(val);names.extend(n);vs.append(video_features(values))
    if use_jepa:
        j=np.asarray(record['jepa'],np.float32)
        if j.ndim!=2 or not len(j) or j.shape[1]<3 or len(record['jepa_names'])!=j.shape[1] or not np.isfinite(j).all():raise ValueError('invalid JEPA schema')
        base=j[:,:3];globals_=video_features(base)
        if temporal:append(base,record['jepa_names'][:3])
        else:
            fs.extend([j,np.broadcast_to(globals_,(len(j),len(globals_))),(base[:,0:1]>0).astype(np.float32)])
            names+=record['jepa_names']+[f'jepa_global_{i}' for i in range(len(globals_))]+['jepa_positive_error_support']
            vs.append(globals_)
    if recipe.startswith('motion_rank'):
        append(record['rank_motion'],['diff_rank','flow_rank','mean_rank','max_rank'])
    if use_motion:append(record['motion'],record['motion_names'])
    if use_corrected:append(record['corrected'],record['corrected_names'])
    if use_rgb:
        if pca is None:raise ValueError('missing train-fold PCA transform')
        rgb=np.asarray(record['rgb'],np.float32);mean=np.asarray(pca['mean'],np.float32);components=np.asarray(pca['components'],np.float32)
        if rgb.ndim!=2 or not len(rgb) or mean.shape!=(rgb.shape[1],) or components.ndim!=2 or components.shape[0]<1 or components.shape[1]!=rgb.shape[1] or not all(np.isfinite(a).all() for a in [rgb,mean,components]):raise ValueError('invalid PCA/RGB schema')
        values=(rgb-mean)@components.T
        append(values,[f'r3d_pca_{i}' for i in range(values.shape[1])])
    if not fs:raise ValueError('unsupported feature recipe: '+recipe)
    if len({len(x) for x in fs})!=1:raise ValueError('unaligned feature arrays')
    return np.concatenate(fs,axis=1).astype(np.float32),np.concatenate(vs).astype(np.float32),names
