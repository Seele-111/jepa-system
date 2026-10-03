"""Compact, FPS-aware, label-free motion/JEPA views for experiment v4.

PCA is explicitly fitted on training content only. This is offline localization;
centered context and future contrast are not claims of causal streaming inference.
"""
from __future__ import annotations
import numpy as np
from optimized_locator import rolling_mean, video_features

KIND = 'compact-fps-context-v4'
RECIPES = ('compact_corrected_motion_et', 'compact_rgb_corrected_motion_et',
           'compact_rgb_corrected_motion_hgb', 'compact_motion_hgb')
TILE_FIELDS = ('residual_p50', 'residual_p90', 'active_fraction', 'speed_mass',
               'active_vx', 'active_vy', 'direction_consistency',
               'acceleration_p90', 'acceleration_mean_x', 'acceleration_mean_y')


def content_weights(records, indices):
    counts = {}
    for i in indices:
        counts[records[i]['sha256']] = counts.get(records[i]['sha256'], 0) + 1
    return {i: 1.0 / counts[records[i]['sha256']] for i in indices}


def fit_rgb_pca(records, indices, components=16):
    """Equal content mass covariance; neither labels nor held-out frames are read."""
    weights = content_weights(records, indices)
    width = records[indices[0]]['rgb'].shape[1]
    mean = np.zeros(width, np.float64)
    second = np.zeros((width, width), np.float64)
    mass = sum(weights.values())
    for i in indices:
        x = np.asarray(records[i]['rgb'], np.float64)
        if x.ndim != 2 or not len(x) or x.shape[1] != width or not np.isfinite(x).all():
            raise ValueError('invalid RGB training features')
        w = weights[i] / (mass * len(x))
        mean += w * x.sum(axis=0)
        second += w * x.T @ x
    cov = second - np.outer(mean, mean)
    eigenvalues, vectors = np.linalg.eigh((cov + cov.T) * .5)
    order = np.argsort(eigenvalues)[::-1][:min(components, width)]
    vectors = vectors[:, order].T
    # Resolve eigenvector sign for repeatable serialization.
    for row in vectors:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return {'mean': mean.astype(np.float32).tolist(),
            'components': vectors.astype(np.float32).tolist(),
            'fit_role': 'training_content_only_equal_content_covariance'}


def _raw(recipe, record, pca):
    continuous, flags, support, cnames, fnames = [], [], [], [], []
    def group(key, flag_indices):
        values = np.asarray(record[key], np.float32)
        names = record[key + '_names']
        if values.ndim != 2 or not len(values) or values.shape[1] != len(names) or not np.isfinite(values).all():
            raise ValueError('invalid raw feature schema: ' + key)
        cc = [i for i in range(len(names)) if i not in flag_indices]
        continuous.append(values[:, cc]); cnames.extend(key + '/' + names[i] for i in cc)
        if key == 'motion':
            valid_indices = [None] * 4 + [22] * 3 + [23] * 2 + [24] * 11 + [25] * 2
        else:
            valid_indices = [6] * 5 + [None] * 2 + [13] * 5 + [None] * 2
        support.append(np.stack([np.ones(len(values), bool) if valid_indices[j] is None
                                  else values[:, valid_indices[j]] > 0 for j in cc], axis=1))
        flags.append(values[:, flag_indices]); fnames.extend(key + '/' + names[i] for i in flag_indices)
    group('motion', [22, 23, 24, 25])
    if 'corrected' in recipe:
        group('corrected', [5, 6, 12, 13])
    values = np.asarray(record['local'], np.float32)
    names = list(record['local_names'])
    if values.ndim != 2 or values.shape[1] != 284 or len(names) != 284 or not np.isfinite(values).all():
        raise ValueError('invalid local compact schema')
    base = {n: j for j, n in enumerate(names[:142])}
    continuous.append(values[:, :32]); cnames.extend(names[:32])
    support.append(values[:, 142:174] > 0)
    flags.append(values[:, 142:174]); fnames.extend(names[142:174])
    for suffix in TILE_FIELDS:
        ids = [base[f'local/tile_{r}{c}_{suffix}'] for r in range(3) for c in range(3)]
        v = values[:, ids]; valid = values[:, np.asarray(ids) + 142]
        if not np.isin(valid, [0, 1]).all():
            raise ValueError('invalid tile support mask')
        count = valid.sum(axis=1)
        mean = (v * valid).sum(axis=1) / np.maximum(1, count)
        var = (((v - mean[:, None]) ** 2) * valid).sum(axis=1) / np.maximum(1, count)
        maximum = np.where(valid > 0, v, -np.inf).max(axis=1)
        maximum[count == 0] = 0
        continuous.append(np.stack([mean, np.sqrt(np.maximum(0, var)), maximum], axis=1))
        support.append(np.repeat((count > 0)[:, None], 3, axis=1))
        cnames.extend(f'local/tile_pool/{suffix}/{stat}' for stat in ['mean', 'std', 'max'])
        flags.append((count / 9)[:, None]); fnames.append(f'local/tile_pool/{suffix}/support_fraction')
    flags.extend([values[:, 131:142], values[:, 273:284]])
    fnames.extend(names[131:142] + names[273:284])
    if 'rgb' in recipe:
        if pca is None:
            raise ValueError('missing training-only RGB PCA')
        x = np.asarray(record['rgb'], np.float32)
        mean = np.asarray(pca['mean'], np.float32)
        components = np.asarray(pca['components'], np.float32)
        if x.ndim != 2 or mean.shape != (x.shape[1],) or components.ndim != 2 or components.shape[1] != x.shape[1] or not len(components) or not all(np.isfinite(z).all() for z in [x, mean, components]):
            raise ValueError('invalid RGB PCA schema')
        continuous.append((x - mean) @ components.T)
        support.append(np.ones((len(x), len(components)), bool))
        cnames.extend(f'rgb/pca_{j}' for j in range(len(components)))
    if len({len(x) for x in continuous + flags}) != 1:
        raise ValueError('unaligned feature frames')
    continuous = np.concatenate(continuous, axis=1).astype(np.float32)
    flags = np.concatenate(flags, axis=1).astype(np.float32)
    if np.any((flags < 0) | (flags > 1)):
        raise ValueError('invalid feature flags')
    return continuous, flags, np.concatenate(support, axis=1), cnames, fnames


def feature_view_v4(recipe, record, pca=None):
    if recipe not in RECIPES:
        raise ValueError('unknown compact recipe')
    fps = float(record['fps'])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('invalid FPS')
    x, flags, support, names, fnames = _raw(recipe, record, pca)
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
    outnames = list(names) + [n + '/' + stat for stat in ['delta_per_second', 'abs_delta_per_second', 'relative_z'] for n in names]
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
