"""v9 fixed normal-negative cost: event-compact representation unchanged."""
from __future__ import annotations
import numpy as np
from optimized_compact_features_v4 import KIND, feature_view_v4, fit_rgb_pca, content_weights
from optimized_compact_model_v4 import export_compact_member
from optimized_event_training_v5 import event_weights

RECIPES=('normalcost_event_compact_rgb_corrected_motion_et',)
VIEW_RECIPES={r:r.removeprefix('normalcost_event_') for r in RECIPES}


def normal_cost_weights(records, train):
    """Double only normal-content negative cost; retain event/background ratios."""
    weights=event_weights(records,train).copy()
    offset=0
    for i in train:
        count=len(records[i]['labels'])
        if not np.any(records[i]['labels']):weights[offset:offset+count]*=2.
        offset+=count
    labels=np.concatenate([records[i]['labels'] for i in train])
    positive,negative=weights[labels>0].sum(),weights[labels==0].sum()
    if min(positive,negative)<=0:raise ValueError('normal-cost fitting requires both classes')
    weights[labels>0]*=negative/positive
    return weights*(len(weights)/weights.sum())


def fit_normal_cost_member(recipe, records, train, predict, seed, *, full_fit=False):
    if recipe not in RECIPES:
        raise ValueError('unsupported v4 recipe')
    if not train or not predict or len(set(train)) != len(train) or len(set(predict)) != len(predict):
        raise ValueError('invalid fit partition')
    if full_fit:
        if set(train) != set(predict):
            raise ValueError('full fit must predict its fit partition explicitly')
    elif {records[i]['sha256'] for i in train} & {records[i]['sha256'] for i in predict}:
        raise ValueError('content leakage')
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    view_recipe = VIEW_RECIPES[recipe]
    pca = fit_rgb_pca(records, train)
    ids = sorted(set(train) | set(predict))
    views = {i: feature_view_v4(view_recipe, records[i], pca) for i in ids}
    names = views[ids[0]][2]
    if any(views[i][2] != names for i in ids):
        raise ValueError('feature names differ')
    x = np.concatenate([views[i][0] for i in train])
    y = np.concatenate([records[i]['labels'] for i in train]).astype(np.uint8)
    content = content_weights(records, train)
    weights = normal_cost_weights(records, train)
    if recipe.endswith('_hgb'):
        model = HistGradientBoostingClassifier(learning_rate=.06, max_iter=180,
            max_leaf_nodes=7, max_depth=3, min_samples_leaf=24, l2_regularization=6,
            max_bins=64, early_stopping=False, max_features=.8, random_state=seed)
    else:
        model = ExtraTreesClassifier(n_estimators=192, max_depth=8,
            min_samples_leaf=16, max_features=.5, n_jobs=2, random_state=seed)
    model.fit(x, y, sample_weight=weights)
    video = np.stack([views[i][1] for i in train])
    vy = np.asarray([int(np.any(records[i]['labels'])) for i in train])
    vw = np.asarray([content[i] * (2 if vy[j] == 0 else 1) for j, i in enumerate(train)])
    vmodel = ExtraTreesClassifier(n_estimators=128, max_depth=4,
        min_samples_leaf=2, max_features=.75, class_weight='balanced', n_jobs=2, random_state=seed + 1100)
    vmodel.fit(video, vy, sample_weight=vw)
    fp = {i: model.predict_proba(views[i][0])[:, 1].astype(np.float32) for i in predict}
    vp = {i: float(vmodel.predict_proba(views[i][1][None, :])[0, 1]) for i in predict}
    state = {'recipe': view_recipe, 'training_recipe_id': recipe, 'transform': {'feature_view': KIND, 'pca': pca,
             'frame_feature_names': names}, 'frame_model': model, 'video_model': vmodel,
             'fit_content_sha256': sorted({records[i]['sha256'] for i in train})}
    return fp, vp, state



def export_normal_cost_member(state):
    result=export_compact_member(state)
    result['training_recipe_id']=state['training_recipe_id']
    return result
