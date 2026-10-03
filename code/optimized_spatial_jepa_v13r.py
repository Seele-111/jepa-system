"""v13r mask-only repair: keep frozen frame features/probabilities unchanged.

Only spatial video statistics are changed to ignore unsupported interpolated values.
The frozen v13 evidence and source are never edited. No production bundle is added.
"""
from __future__ import annotations
import numpy as np
from optimized_spatial_jepa_v13 import feature_view_spatial,RECIPE,BASE_RECIPE
from optimized_feature_blocks_v10 import feature_view_blocks
from optimized_compact_features_v4 import content_weights
from optimized_locator import video_features,export_model,portable_predict

KIND='spatial-jepa-mask-repair-v13r'


def masked_video_statistics(values,support):
    values=np.asarray(values,np.float32);mask=np.asarray(support,bool)
    if values.ndim!=2 or mask.shape!=values.shape or not len(values) or not np.isfinite(values[mask]).all():
        raise ValueError('invalid masked spatial video input')
    result=np.zeros((4,values.shape[1]),np.float32)
    for j in range(values.shape[1]):
        observed=values[mask[:,j],j]
        if len(observed):result[:,j]=[observed.mean(),observed.std(),np.percentile(observed,10),np.percentile(observed,90)]
    return result.reshape(-1)


def feature_view_spatial_repaired(record):
    frame,_,names=feature_view_spatial(RECIPE,record,None)
    _,base,_=feature_view_blocks(BASE_RECIPE,record,None)
    extra=masked_video_statistics(record['spatial'],record['spatial_support'])
    video=np.concatenate([base,extra,video_features(record['spatial_support'].astype(np.float32))]).astype(np.float32)
    if not np.isfinite(video).all():raise ValueError('invalid repaired spatial video view')
    return frame,video,names


def fit_repaired_video(records,train,predict,seed):
    if not train or not predict or len(set(train))!=len(train) or len(set(predict))!=len(predict):raise ValueError('invalid repaired video partition')
    if {records[i]['sha256'] for i in train}&{records[i]['sha256'] for i in predict}:raise ValueError('repaired video content leakage')
    from sklearn.ensemble import ExtraTreesClassifier
    ids=sorted(set(train)|set(predict));views={i:feature_view_spatial_repaired(records[i]) for i in ids}
    video=np.stack([views[i][1] for i in train]);y=np.asarray([int(np.any(records[i]['labels'])) for i in train]);cw=content_weights(records,train)
    weights=np.asarray([cw[i]*(2 if y[j]==0 else 1) for j,i in enumerate(train)])
    model=ExtraTreesClassifier(n_estimators=128,max_depth=4,min_samples_leaf=2,max_features=.75,class_weight='balanced',n_jobs=2,random_state=seed+1100)
    model.fit(video,y,sample_weight=weights);portable=export_model(model);vp={};error=0.
    for i in predict:
        ids1=np.flatnonzero(model.classes_==1)
        expected=float(model.predict_proba(views[i][1][None])[0,int(ids1[0])]) if len(ids1) else 0.
        actual=float(portable_predict(portable,views[i][1][None])[0]);error=max(error,abs(actual-expected));vp[i]=expected
    if error>2e-6:raise ValueError('repaired video portable parity differs')
    transform={'feature_view':KIND,'pca':None,'frame_feature_names':views[ids[0]][2],
        'video_statistics':'spatial_support_masked_mean_std_p10_p90_empty_zero_and_support_shape'}
    return vp,{'fit_content_sha256':sorted({records[i]['sha256'] for i in train}),'transform':transform,'max_portable_error':error,'model_sha256':__import__('hashlib').sha256(__import__('json').dumps(portable,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()}
