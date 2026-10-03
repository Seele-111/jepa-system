"""Training and portable boosted trees for the bounded v4 experiment."""
from __future__ import annotations
import numpy as np
from optimized_compact_features_v4 import RECIPES, KIND, feature_view_v4, fit_rgb_pca, content_weights
from optimized_locator import export_model, portable_predict


def fit_compact_member(recipe, records, train, predict, seed, *, full_fit=False):
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
    pca = fit_rgb_pca(records, train) if 'rgb' in recipe else None
    ids = sorted(set(train) | set(predict))
    views = {i: feature_view_v4(recipe, records[i], pca) for i in ids}
    names = views[ids[0]][2]
    if any(views[i][2] != names for i in ids):
        raise ValueError('feature names differ')
    x = np.concatenate([views[i][0] for i in train])
    y = np.concatenate([records[i]['labels'] for i in train]).astype(np.uint8)
    content = content_weights(records, train)
    weights = np.concatenate([np.full(len(records[i]['labels']), content[i] * (2 if not np.any(records[i]['labels']) else 1) / len(records[i]['labels'])) for i in train])
    positive, negative = weights[y > 0].sum(), weights[y == 0].sum()
    if min(positive, negative) <= 0:
        raise ValueError('training requires both frame classes')
    weights[y > 0] *= negative / positive
    weights *= len(weights) / weights.sum()
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
    state = {'recipe': recipe, 'transform': {'feature_view': KIND, 'pca': pca,
             'frame_feature_names': names}, 'frame_model': model, 'video_model': vmodel,
             'fit_content_sha256': sorted({records[i]['sha256'] for i in train})}
    return fp, vp, state


def export_hgb(model):
    if len(model.classes_) != 2 or list(model.classes_) != [0, 1]:
        raise ValueError('portable HGB requires binary [0,1] class ordering')
    if getattr(model, 'is_categorical_', None) is not None and np.any(model.is_categorical_):
        raise ValueError('categorical HGB is not supported')
    trees = []
    for predictors in model._predictors:
        if len(predictors) != 1:
            raise ValueError('portable HGB requires binary logits')
        n = predictors[0].nodes
        if np.any(n['is_categorical']):
            raise ValueError('categorical HGB is not supported')
        trees.append({key: n[field].tolist() for key, field in [
            ('left', 'left'), ('right', 'right'), ('feature', 'feature_idx'),
            ('threshold', 'num_threshold'), ('leaf', 'is_leaf'), ('value', 'value')]})
    return {'kind': 'hist-gradient-boosting-logit-v4',
            'n_features': int(model.n_features_in_),
            'bias': float(model._baseline_prediction.ravel()[0]), 'trees': trees}


def portable_hgb(model, values):
    if model.get('kind') != 'hist-gradient-boosting-logit-v4':
        raise ValueError('unknown boosted model')
    x = np.asarray(values, np.float32)
    if x.ndim != 2 or x.shape[1] != model['n_features'] or not np.isfinite(x).all():
        raise ValueError('invalid boosted model features')
    logit = np.full(len(x), model['bias'], np.float64)
    for tree in model['trees']:
        node = np.zeros(len(x), np.int64)
        leaf = np.asarray(tree['leaf'], bool); feature = np.asarray(tree['feature'], np.int64)
        threshold = np.asarray(tree['threshold'], np.float64)
        left, right = np.asarray(tree['left'], np.int64), np.asarray(tree['right'], np.int64)
        value = np.asarray(tree['value'], np.float64)
        for _ in range(len(leaf)):
            active = np.flatnonzero(~leaf[node])
            if not len(active):
                break
            current = node[active]
            node[active] = np.where(x[active, feature[current]] <= threshold[current], left[current], right[current])
        else:
            raise ValueError('invalid boosted tree topology')
        logit += value[node]
    return (1 / (1 + np.exp(-np.clip(logit, -700, 700)))).astype(np.float32)


def export_compact_member(state):
    fm = export_hgb(state['frame_model']) if state['recipe'].endswith('_hgb') else export_model(state['frame_model'])
    return {'recipe': state['recipe'], 'weight': 1., 'transform': state['transform'],
            'frame_model': fm, 'video_model': export_model(state['video_model']), 'boundary_models': None}


def predict_compact_member(member, record):
    x, v, names = feature_view_v4(member['recipe'], record, member['transform'].get('pca'))
    if names != member['transform']['frame_feature_names']:
        raise ValueError('compact inference feature names differ')
    model = member['frame_model']
    fp = portable_hgb(model, x) if model.get('kind') == 'hist-gradient-boosting-logit-v4' else portable_predict(model, x)
    vp = float(portable_predict(member['video_model'], v[None, :])[0])
    return fp, vp
