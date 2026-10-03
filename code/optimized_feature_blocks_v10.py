"""Controlled feature-block ablation; frozen v4 transform and v8 ET objective.

No label or identity enters the transform. Motion includes local camera-compensated
signals. JEPA means the verified corrected predictor/target features, not proxies.
The copied transform preserves the frozen v4 implementation rather than editing it.
"""
from __future__ import annotations
import numpy as np
from optimized_compact_features_v4 import _raw, content_weights, fit_rgb_pca
from optimized_locator import rolling_mean, video_features, export_model, portable_predict
from optimized_event_training_v5 import event_weights

KIND = 'compact-feature-blocks-v10'
RECIPES = ('blocks_motion_et', 'blocks_rgb_motion_et',
           'blocks_corrected_motion_et', 'blocks_rgb_corrected_motion_et')
BASE_RECIPES = {r: r.replace('blocks_', 'compact_') for r in RECIPES}


def feature_view_blocks(recipe, record, pca=None):
    if recipe not in RECIPES:
        raise ValueError('unknown feature-block recipe')
    fps = float(record['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('invalid FPS')
    x, flags, support, names, fnames = _raw(BASE_RECIPES[recipe], record, pca)
    x = np.where(support, x, 0).astype(np.float32)
    delta = np.vstack([np.zeros((1, x.shape[1]), np.float32), np.diff(x, axis=0)]) * fps
    pair_support = support & np.vstack([np.zeros((1, x.shape[1]), bool), support[:-1]])
    delta = np.where(pair_support, delta, 0)
    z = np.zeros_like(x)
    for j in range(x.shape[1]):
        observed = x[support[:, j], j]
        if not len(observed):
            continue
        q = np.percentile(observed, [25, 75])
        scale = max((q[1] - q[0]) / 1.349, np.std(observed) * .1 + 1e-5)
        z[support[:, j], j] = np.clip((observed - np.median(observed)) / scale, -8, 8)
    features = [x, delta, np.abs(delta), z]
    outnames = list(names) + [n + '/' + stat for stat in
        ['delta_per_second', 'abs_delta_per_second', 'relative_z'] for n in names]
    for seconds in [.2, .6]:
        width = max(1, int(round(seconds * fps)))
        mass = rolling_mean(support.astype(np.float32), width)
        mean = np.divide(rolling_mean(x, width), mass, out=np.zeros_like(x), where=mass > 0)
        features.extend([mean, np.where(support & (mass > 0), x - mean, 0)])
        outnames.extend(n + f'/{stat}_{seconds}s' for stat in ['mean', 'contrast'] for n in names)
    width = max(1, int(round(.3 * fps)))
    def side_means(values):
        padded = np.pad(values, [(width, width), (0, 0)], mode='edge').astype(np.float64)
        cs = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(padded, axis=0)])
        return ((cs[width:width + len(x)] - cs[:len(x)]) / width,
                (cs[2 * width + 1:2 * width + 1 + len(x)] - cs[width + 1:width + 1 + len(x)]) / width)
    past, future = side_means(x)
    past_mass, future_mass = side_means(support)
    past = np.divide(past, past_mass, out=np.zeros_like(past), where=past_mass > 0)
    future = np.divide(future, future_mass, out=np.zeros_like(future), where=future_mass > 0)
    contrast = np.where((past_mass > 0) & (future_mass > 0), future - past, 0)
    features.append(contrast.astype(np.float32))
    outnames.extend(n + '/future_minus_past_0.3s' for n in names)
    features.append(flags); outnames.extend(fnames)
    result = np.concatenate(features, axis=1).astype(np.float32)
    video = np.concatenate([video_features(x), video_features(flags)]).astype(np.float32)
    if not np.isfinite(result).all() or result.shape[1] != len(outnames):
        raise ValueError('invalid compact transform output')
    return result, video, outnames


def fit_block_member(recipe, records, train, predict, seed, *, full_fit=False):
    if recipe not in RECIPES or not train or not predict:
        raise ValueError('invalid recipe or empty partition')
    if len(set(train)) != len(train) or len(set(predict)) != len(predict):
        raise ValueError('duplicate partition index')
    if full_fit:
        if set(train) != set(predict):
            raise ValueError('full fit must predict exactly its fit partition')
    elif {records[i]['sha256'] for i in train} & {records[i]['sha256'] for i in predict}:
        raise ValueError('content leakage')
    from sklearn.ensemble import ExtraTreesClassifier
    pca = fit_rgb_pca(records, train) if 'rgb' in recipe else None
    ids = sorted(set(train) | set(predict))
    views = {i: feature_view_blocks(recipe, records[i], pca) for i in ids}
    names = views[ids[0]][2]
    if any(views[i][2] != names for i in ids):
        raise ValueError('feature names differ')
    x = np.concatenate([views[i][0] for i in train])
    y = np.concatenate([records[i]['labels'] for i in train]).astype(np.uint8)
    weights = event_weights(records, train)
    model = ExtraTreesClassifier(n_estimators=192, max_depth=8,
        min_samples_leaf=16, max_features=.5, n_jobs=2, random_state=seed)
    model.fit(x, y, sample_weight=weights)
    video = np.stack([views[i][1] for i in train])
    vy = np.asarray([int(np.any(records[i]['labels'])) for i in train])
    cw = content_weights(records, train)
    vw = np.asarray([cw[i] * (2 if vy[j] == 0 else 1) for j, i in enumerate(train)])
    vm = ExtraTreesClassifier(n_estimators=128, max_depth=4, min_samples_leaf=2,
        max_features=.75, class_weight='balanced', n_jobs=2, random_state=seed + 1100)
    vm.fit(video, vy, sample_weight=vw)
    def positive(m, values):
        ids = np.flatnonzero(m.classes_ == 1)
        return m.predict_proba(values)[:, int(ids[0])].astype(np.float32) if len(ids) else np.zeros(len(values), np.float32)
    fp = {i: positive(model, views[i][0]) for i in predict}
    vp = {i: float(positive(vm, views[i][1][None])[0]) for i in predict}
    state = {'recipe': recipe, 'transform': {'feature_view': KIND, 'pca': pca,
        'frame_feature_names': names}, 'frame_model': model, 'video_model': vm,
        'fit_content_sha256': sorted({records[i]['sha256'] for i in train})}
    return fp, vp, state


def export_block_member(state):
    return {'recipe': state['recipe'], 'transform': state['transform'],
        'frame_model': export_model(state['frame_model']),
        'video_model': export_model(state['video_model']),
        'fit_content_sha256': state['fit_content_sha256']}


def predict_block_member(member, record):
    x, v, names = feature_view_blocks(member['recipe'], record, member['transform']['pca'])
    if names != member['transform']['frame_feature_names']:
        raise ValueError('trained/inference feature order mismatch')
    return portable_predict(member['frame_model'], x), float(portable_predict(member['video_model'], v[None])[0])
