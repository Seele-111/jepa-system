"""Label-free v2 feature view: temporal frames and video identity context separate."""
from __future__ import annotations
import numpy as np
from optimized_locator import augment_signals, video_features

KIND = 'shared-label-free-v2'


def feature_view_v2(recipe, record, pca=None):
    fps = float(record['fps'])
    if not np.isfinite(fps) or fps<=0: raise ValueError('invalid FPS')
    groups=[]
    if 'motion' in recipe: groups.append((record['motion'],record['motion_names']))
    if 'corrected' in recipe: groups.append((record['corrected'],record['corrected_names']))
    if 'local' in recipe: groups.append((record['local'],record['local_names']))
    if recipe.startswith('rgb_'):
        if not pca: raise ValueError('missing train-fold PCA')
        rgb = np.asarray(record['rgb'],np.float32)
        mean=np.asarray(pca['mean'],np.float32);components=np.asarray(pca['components'],np.float32)
        if rgb.ndim!=2 or mean.shape!=(rgb.shape[1],) or components.ndim!=2 or not len(components) or components.shape[1]!=rgb.shape[1] or not all(np.isfinite(v).all() for v in [rgb,mean,components]): raise ValueError('invalid PCA schema')
        groups.append(((rgb-mean)@components.T,[f'r3d_pca_{i}' for i in range(len(components))]))
    if not groups: raise ValueError('unsupported v2 recipe')
    fs=[];vs=[];names=[]
    for values,raw in groups:
        values=np.asarray(values,np.float32);raw=list(raw)
        if values.ndim!=2 or not len(values) or values.shape[1]!=len(raw) or not np.isfinite(values).all(): raise ValueError('invalid v2 raw schema')
        flags=[i for i,n in enumerate(raw) if n.endswith('_valid') or 'direct_observation' in n or 'interpolation_range_supported' in n or n.endswith('_support_flag') or n.split('/')[-1] in ('camera_fallback_used','texture_observable','source_size_changed','flow_failed','camera_model_changed')]
        continuous=[i for i in range(len(raw)) if i not in flags]
        if continuous:
            part,part_names=augment_signals(values[:,continuous],[raw[i] for i in continuous],fps)
            stop=part.shape[1]-4*len(continuous)
            fs.append(part[:,:stop]);names.extend(part_names[:stop])
        if flags:
            if np.any((values[:,flags]<0)|(values[:,flags]>1)): raise ValueError('invalid validity flag')
            fs.append(values[:,flags]);names.extend(raw[i] for i in flags)
        vs.append(video_features(values))
    if len({len(x) for x in fs})!=1: raise ValueError('unaligned v2 arrays')
    return np.concatenate(fs,axis=1).astype(np.float32),np.concatenate(vs).astype(np.float32),names
