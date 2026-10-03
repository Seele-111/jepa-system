"""Event-balanced (not merely class-balanced) fitting for experiment v5."""
from __future__ import annotations
import numpy as np
from optimized_compact_features_v4 import fit_rgb_pca, content_weights
from optimized_compact_model_v4 import export_hgb, portable_hgb
from optimized_feature_view import feature_view
from optimized_locator import spans, export_model, portable_predict

RECIPES = ('event_corrected_motion_et', 'event_rgb_corrected_motion_et', 'event_rgb_corrected_motion_hgb')
VIEW_RECIPES = {'event_corrected_motion_et': 'corrected_motion_et',
                'event_rgb_corrected_motion_et': 'rgb_corrected_motion_rf',
                'event_rgb_corrected_motion_hgb': 'rgb_corrected_motion_rf'}


def event_weights(records, train):
    """Equal positive/negative mass per content; events get equal positive mass.

    Normal content has negative mass 2; anomalous content has positive mass 1,
    split equally across inclusive events, and background mass 1 if present.
    Finally balance total positive and negative mass and normalize mean to one.
    """
    content = content_weights(records, train)
    rows = []
    for i in train:
        labels = np.asarray(records[i]['labels'], np.uint8)
        events = spans(labels > 0)
        w = np.zeros(len(labels), np.float64)
        negative = labels == 0
        if np.any(negative):
            w[negative] = content[i] * (2 if not events else 1) / negative.sum()
        if events:
            for s, e in events:
                w[s:e + 1] = content[i] / (len(events) * (e - s + 1))
        rows.append(w)
    weights = np.concatenate(rows); labels = np.concatenate([records[i]['labels'] for i in train])
    positive, negative = weights[labels > 0].sum(), weights[labels == 0].sum()
    if min(positive, negative) <= 0:
        raise ValueError('event fitting requires both frame classes')
    weights[labels > 0] *= negative / positive
    return weights * (len(weights) / weights.sum())


def fit_event_member(recipe, records, train, predict, seed, *, full_fit=False):
    if recipe not in RECIPES or not train or not predict:
        raise ValueError('unknown event recipe or empty partition')
    if len(set(train)) != len(train) or len(set(predict)) != len(predict):
        raise ValueError('duplicate row index')
    if full_fit:
        if set(train) != set(predict): raise ValueError('full fit partition mismatch')
    elif {records[i]['sha256'] for i in train} & {records[i]['sha256'] for i in predict}:
        raise ValueError('content leakage')
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    view_recipe = VIEW_RECIPES[recipe]
    pca = fit_rgb_pca(records, train) if 'rgb' in recipe else None
    views = {i: feature_view(view_recipe, records[i], pca) for i in sorted(set(train) | set(predict))}
    names = views[train[0]][2]
    if any(v[2] != names for v in views.values()): raise ValueError('feature name mismatch')
    x = np.concatenate([views[i][0] for i in train]); y = np.concatenate([records[i]['labels'] for i in train])
    weights = event_weights(records, train)
    if recipe.endswith('_hgb'):
        model = HistGradientBoostingClassifier(learning_rate=.05, max_iter=200, max_leaf_nodes=7,
            max_depth=3, min_samples_leaf=20, l2_regularization=8, max_bins=64,
            max_features=.8, early_stopping=False, random_state=seed)
    else:
        model = ExtraTreesClassifier(n_estimators=224, max_depth=9, min_samples_leaf=8,
                                     max_features=.6, n_jobs=2, random_state=seed)
    model.fit(x, y, sample_weight=weights)
    video = np.stack([views[i][1] for i in train]); vy = np.asarray([int(np.any(records[i]['labels'])) for i in train])
    content = content_weights(records, train)
    video_weights = np.array([content[i] for i in train])
    vmodel = ExtraTreesClassifier(n_estimators=128, max_depth=4, min_samples_leaf=2,
                                 max_features=.75, class_weight='balanced', n_jobs=2, random_state=seed + 1100)
    vmodel.fit(video, vy, sample_weight=video_weights)
    fp = {i: model.predict_proba(views[i][0])[:, 1].astype(np.float32) for i in predict}
    vp = {i: float(vmodel.predict_proba(views[i][1][None])[0, 1]) for i in predict}
    state = {'recipe': view_recipe, 'training_recipe_id': recipe,
             'transform': {'recipe': view_recipe, 'frame_feature_names': names, 'pca': pca, 'feature_view': 'shared-label-free-v1'},
             'frame_model': model, 'video_model': vmodel,
             'fit_content_sha256': sorted({records[i]['sha256'] for i in train})}
    return fp, vp, state


def export_event_member(state):
    fm = export_hgb(state['frame_model']) if state['training_recipe_id'].endswith('_hgb') else export_model(state['frame_model'])
    return {'recipe': state['recipe'], 'training_recipe_id': state['training_recipe_id'],
            'transform': state['transform'], 'weight': 1., 'frame_model': fm,
            'video_model': export_model(state['video_model']), 'boundary_models': None}


def predict_event_member(member, record):
    x, v, names = feature_view(member['recipe'], record, member['transform'].get('pca'))
    if names != member['transform']['frame_feature_names']: raise ValueError('event feature names differ')
    m = member['frame_model']
    fp = portable_hgb(m, x) if m.get('kind') == 'hist-gradient-boosting-logit-v4' else portable_predict(m, x)
    return fp, float(portable_predict(member['video_model'], v[None])[0])
