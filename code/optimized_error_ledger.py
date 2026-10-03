"""Read-only ledger for frozen, content-held-out nested predictions.

No model/decoder imports, inference, fitting, extraction or network calls. Event
metrics deliberately use the project's prediction-order greedy inclusive IoU,
not maximum-cardinality/Hungarian matching. Split/merge and boundary flags are
geometric diagnostic proxies; every semantic cause stays unknown.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from numbers import Integral
import os
from pathlib import Path
import re
import statistics

import numpy as np

SCHEMA = 'optimization-oof-error-ledger-v1'
ORIGINAL = Path('output/algorithm-opt-2026-10-02')
GROUPED = Path('output/algorithm-opt-2026-10-02-v2')
BASELINE_REPORT = GROUPED / 'selection_audited/report.json'
VERSIONS = ('v1', 'v4', 'v5', 'v8', 'v9')
REPORT_SCHEMAS = {'v1': 'algorithm-opt-grouped-recall-v2',
                  'v4': 'compact-experiment-v4', 'v5': 'event-experiment-v5',
                  'v8': 'event-compact-crossed-v8', 'v9': 'normal-negative-cost-v9'}
EXPERIMENTS = {v: Path('output') / ('algorithm-opt-2026-10-02-' + v)
               for v in VERSIONS if v != 'v1'}
RECIPE_OWNERS = {r: 'v1' for r in (
    'motion_rf', 'motion_et', 'rgb_motion_tcn', 'corrected_motion_rf',
    'corrected_motion_et', 'corrected_motion_tcn', 'rgb_corrected_motion_rf',
    'corrected_motion_boundary_rf', 'corrected_motion_boundary_et')}
RECIPE_OWNERS.update({r: 'v4' for r in (
    'compact_corrected_motion_et', 'compact_rgb_corrected_motion_et',
    'compact_rgb_corrected_motion_hgb', 'compact_motion_hgb')})
RECIPE_OWNERS.update({r: 'v5' for r in (
    'event_corrected_motion_et', 'event_rgb_corrected_motion_et',
    'event_rgb_corrected_motion_hgb')})
RECIPE_OWNERS.update({r: 'v8' for r in (
    'event_compact_rgb_corrected_motion_et', 'event_compact_rgb_corrected_motion_hgb')})
RECIPE_OWNERS['normalcost_event_compact_rgb_corrected_motion_et'] = 'v9'
THRESHOLDS = (.3, .5)


class LedgerError(ValueError):
    """Missing, inconsistent or non-OOF evidence; never silently substitute."""


def require(condition, message):
    if not condition:
        raise LedgerError(message)


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(
        c in '0123456789abcdef' for c in value)


def checked_fps(fps):
    require(not isinstance(fps, bool) and isinstance(fps, (int, float)) and
            math.isfinite(fps) and fps > 0, 'FPS must be finite and positive')
    return float(fps)


def checked_labels(labels):
    values = np.asarray(labels)
    require(values.ndim == 1 and len(values) > 0 and values.dtype.kind in 'biuf' and
            np.isin(values, [0, 1]).all(), 'labels must be a nonempty binary vector')
    return values.astype(bool, copy=False)


def checked_intervals(intervals, frames=None):
    require(isinstance(intervals, (list, tuple)), 'intervals must be a list/tuple')
    result = []
    for span in intervals:
        require(isinstance(span, (list, tuple)) and len(span) == 2 and all(
            isinstance(x, Integral) and not isinstance(x, (bool, np.bool_)) for x in span),
            'interval endpoints must be integers')
        start, end = map(int, span)
        require(0 <= start <= end and (frames is None or end < frames),
                'interval outside the original frame grid')
        result.append((start, end))
    return result  # Preserve prediction order, duplicates and overlap.


def label_spans(labels):
    result, start = [], None
    values = checked_labels(labels)
    for frame, positive in enumerate(values):
        if positive and start is None:
            start = frame
        elif not positive and start is not None:
            result.append((start, frame - 1))
            start = None
    if start is not None:
        result.append((start, len(values) - 1))
    return result


def intersection_frames(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def event_iou(a, b):
    intersection = intersection_frames(a, b)
    return intersection / (a[1] - a[0] + b[1] - b[0] + 2 - intersection)


def match_events(predictions, truth, threshold):
    """One prediction/one GT at most; strict > tie update, >= threshold accept."""
    pred, gt = checked_intervals(predictions), checked_intervals(truth)
    require(not isinstance(threshold, bool) and isinstance(threshold, (int, float))
            and math.isfinite(threshold) and 0 < threshold <= 1, 'invalid IoU threshold')
    matches, used = [], set()
    for prediction_id, span in enumerate(pred):
        best_id, best_overlap = None, 0.
        for event_id, event in enumerate(gt):
            if event_id in used:
                continue
            overlap = event_iou(span, event)
            if overlap > best_overlap:
                best_id, best_overlap = event_id, overlap
        if best_id is not None and best_overlap >= threshold:
            used.add(best_id)
            matches.append({'prediction_id': prediction_id, 'event_id': best_id,
                            'iou': best_overlap})
    predicted = {m['prediction_id'] for m in matches}
    return {'threshold': threshold, 'matches': matches, 'tp': len(matches),
            'fp': len(pred) - len(matches), 'fn': len(gt) - len(matches),
            'unmatched_prediction_ids': [i for i in range(len(pred)) if i not in predicted],
            'unmatched_event_ids': [i for i in range(len(gt)) if i not in used]}


def timed_interval(span, fps):
    start, end = checked_intervals([span])[0]
    fps = checked_fps(fps)
    return {'start_frame': start, 'end_frame_inclusive': end,
            'start_seconds': start / fps, 'end_seconds_exclusive': (end + 1) / fps,
            'duration_frames': end - start + 1, 'duration_seconds': (end - start + 1) / fps}


def boundary_offsets(prediction, truth, fps):
    pred, gt = checked_intervals([prediction, truth])
    fps = checked_fps(fps)
    return {'start_offset_frames': pred[0] - gt[0],
            'end_offset_frames': pred[1] - gt[1],
            'start_offset_seconds': (pred[0] - gt[0]) / fps,
            'end_offset_seconds': (pred[1] - gt[1]) / fps}


def count_metrics(tp, fp, fn):
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp / max(1, tp + fp),
            'recall': tp / max(1, tp + fn), 'f1': 2 * tp / max(1, 2 * tp + fp + fn)}


def analyze_video(record, predictions, pipeline='v8', provenance=None):
    """Return a video row, GT event rows and candidate rows without guessing causes."""
    labels, fps = checked_labels(record['labels']), checked_fps(record['fps'])
    pred = checked_intervals(predictions, len(labels))
    truth = label_spans(labels)
    matches = {f'iou_{t:.1f}': match_events(pred, truth, t) for t in THRESHOLDS}
    event_to_pred = [[j for j, p in enumerate(pred) if intersection_frames(g, p) > 0]
                     for g in truth]
    pred_to_event = [[j for j, g in enumerate(truth) if intersection_frames(g, p) > 0]
                     for p in pred]
    shared = {'pipeline': pipeline, 'index': record['index'], 'name': record['name'],
              'sha256': record['sha256'], 'fold': (provenance or {}).get('fold'),
              'fps': fps, 'semantic_cause': 'unknown'}
    errors, events, candidates = [], [], []
    if not truth and pred:
        errors.append('normal_false_positive')
    if truth and not pred:
        errors.append('abnormal_without_candidate')
    for event_id, span in enumerate(truth):
        overlaps = event_to_pred[event_id]
        flags = []
        row = {**shared, 'event_id': event_id, **timed_interval(span, fps),
               'overlapping_prediction_ids': overlaps, 'split_proxy': len(overlaps) > 1,
               'merge_proxy_prediction_ids': [i for i in overlaps if len(pred_to_event[i]) > 1],
               'diagnostic_only': True}
        for key, result in matches.items():
            match = next((m for m in result['matches'] if m['event_id'] == event_id), None)
            row['match_' + key] = match
            row['boundary_' + key] = (boundary_offsets(pred[match['prediction_id']], span, fps)
                                      if match else None)
            if not match:
                flags.append('event_miss_at_' + key)
            elif any(row['boundary_' + key][k] != 0 for k in
                     ('start_offset_frames', 'end_offset_frames')):
                flags.append('boundary_offset_at_' + key)
        if not overlaps:
            flags.append('no_overlapping_candidate')
        if row['split_proxy']:
            flags.append('split_proxy')
        if row['merge_proxy_prediction_ids']:
            flags.append('merge_proxy')
        best = max(overlaps, key=lambda i: event_iou(span, pred[i])) if overlaps else None
        row['best_overlap_diagnostic'] = (None if best is None else {
            'prediction_id': best, 'iou': event_iou(span, pred[best]),
            'not_an_additional_match': True, **boundary_offsets(pred[best], span, fps)})
        row['abnormal_without_candidate'] = not bool(pred)
        row['error_types'] = flags
        errors.extend(flags)
        events.append(row)
    for prediction_id, span in enumerate(pred):
        flags = []
        row = {**shared, 'prediction_id': prediction_id, **timed_interval(span, fps),
               'overlapping_event_ids': pred_to_event[prediction_id],
               'merge_proxy': len(pred_to_event[prediction_id]) > 1, 'diagnostic_only': True}
        for key, result in matches.items():
            match = next((m for m in result['matches'] if m['prediction_id'] == prediction_id), None)
            row['match_' + key] = match
            if not match:
                flags.append('unmatched_prediction_at_' + key)
        if not truth:
            flags.append('normal_false_positive')
        if not pred_to_event[prediction_id]:
            flags.append('no_overlapping_event')
        if row['merge_proxy']:
            flags.append('merge_proxy')
        row['error_types'] = flags
        errors.extend(flags)
        candidates.append(row)
    mask = np.zeros(len(labels), dtype=bool)
    for start, end in pred:
        mask[start:end + 1] = True
    frame = count_metrics(int(np.count_nonzero(mask & labels)),
                          int(np.count_nonzero(mask & ~labels)),
                          int(np.count_nonzero(~mask & labels)))
    video = {**shared, 'frames': len(labels), 'duration_seconds': len(labels) / fps,
             'event_count': len(truth), 'prediction_count': len(pred),
             'is_normal': not bool(truth), 'normal_false_positive': not truth and bool(pred),
             'abnormal_without_candidate': bool(truth) and not pred,
             'split_proxy_event_ids': [i for i, p in enumerate(event_to_pred) if len(p) > 1],
             'merge_proxy_prediction_ids': [i for i, g in enumerate(pred_to_event) if len(g) > 1],
             'ground_truth_events': [{'event_id': i, **timed_interval(s, fps)}
                                     for i, s in enumerate(truth)],
             'predicted_segments': [{'prediction_id': i, **timed_interval(s, fps)}
                                    for i, s in enumerate(pred)],
             'event_matching': matches, 'frame': frame, 'error_types': sorted(set(errors)),
             'oof_provenance': provenance or {}}
    # Ensure flags are actual booleans, not empty lists (Python and/or semantics).
    video['normal_false_positive'] = bool(video['normal_false_positive'])
    video['abnormal_without_candidate'] = bool(video['abnormal_without_candidate'])
    return video, events, candidates


def alias_conflicts(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['sha256']].append(row)
    result = []
    for digest, rows in groups.items():
        if len(rows) < 2:
            continue
        labels = [checked_labels(r['labels']) for r in rows]
        comparable = len({(len(y), checked_fps(r['fps'])) for r, y in zip(rows, labels)}) == 1
        conflicts = ([i for i in range(len(labels[0])) if len({int(y[i]) for y in labels}) > 1]
                     if comparable else None)
        result.append({'sha256': digest, 'indices': [r['index'] for r in rows],
                       'names': [r['name'] for r in rows], 'metadata_comparable': comparable,
                       'label_conflict': bool(conflicts) if comparable else None,
                       'conflicting_frame_indices': conflicts,
                       'conflicting_frames': ([{'frame': i, 'frame_time_seconds': i / rows[0]['fps'],
                           'original_labels_by_index': {str(r['index']): int(y[i])
                                                       for r, y in zip(rows, labels)}}
                                              for i in conflicts] if comparable else []),
                       'original_events_by_index': {str(r['index']): [list(s) for s in label_spans(y)]
                                                    for r, y in zip(rows, labels)},
                       'resolution': 'original_annotations_preserved_no_relabeling',
                       'semantic_cause': 'unknown'})
    return result


class InputEvidence:
    """Hash the bytes actually read; only workspace-local fixed paths are allowed."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.input_hashes = {}
        self.receipts = {}

    def read(self, relative, expected=None):
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root), 'input path escapes workspace')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        key = path.relative_to(self.root).as_posix()
        if expected is not None:
            require(valid_digest(expected) and digest == expected, 'input hash mismatch: ' + key)
        require(key not in self.input_hashes or self.input_hashes[key] == digest,
                'input changed during ledger construction: ' + key)
        self.input_hashes[key] = digest
        return data

    def json(self, relative, expected=None):
        try:
            return json.loads(self.read(relative, expected).decode('utf-8'),
                              parse_constant=lambda x: self.reject_constant(x))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise LedgerError('invalid JSON: ' + str(relative)) from exc

    @staticmethod
    def reject_constant(value):
        raise LedgerError('nonfinite JSON constant: ' + value)

    def hash(self, relative):
        self.read(relative)
        return self.input_hashes[Path(relative).as_posix()]

    def receipt(self, version):
        if version not in self.receipts:
            folder = EXPERIMENTS[version]
            receipt = self.json(folder / 'training_receipt.json')
            require(value_hash({k: v for k, v in receipt.items() if k != 'receipt_sha256'}) ==
                    receipt.get('receipt_sha256'), 'training receipt signature mismatch')
            pins = receipt.get('input_hashes', {})
            for path in (ORIGINAL / 'dataset_manifest.json', ORIGINAL / 'dataset.npz',
                         GROUPED / 'dataset_manifest.json'):
                require(path.as_posix() in pins, 'receipt missing original-label/FPS binding')
                self.read(path, pins[path.as_posix()])
            self.read(BASELINE_REPORT, receipt['baseline_report_sha256'])
            self.read(folder / 'protocol.json', receipt['protocol_sha256'])
            self.receipts[version] = receipt
        return self.receipts[version]


def load_original_records(evidence):
    original = evidence.json(ORIGINAL / 'dataset_manifest.json')
    manifest = evidence.json(GROUPED / 'dataset_manifest.json')
    require(original.get('schema_version') == 'algorithm-opt-dataset-v1' and
            manifest.get('schema_version') == 'algorithm-opt-grouped-dataset-v2',
            'unsupported original/grouped dataset schema')
    require(manifest.get('rows') == original.get('rows') and
            manifest.get('v1_manifest_sha256') == evidence.hash(ORIGINAL / 'dataset_manifest.json'),
            'original rows/manifest binding differs')
    require(len(manifest['rows']) == 77, 'formal ledger requires the original 77 rows')
    data = evidence.read(ORIGINAL / 'dataset.npz')
    records = []
    with np.load(BytesIO(data), allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), 'duplicate dataset NPZ keys')
        require({f'labels_{i}' for i in range(77)} <= set(archive.files), 'original labels missing')
        for index, row in enumerate(manifest['rows']):
            require(valid_digest(row.get('sha256')) and isinstance(row.get('name'), str) and
                    type(row.get('frames')) is int and row['frames'] > 0,
                    'invalid original row metadata')
            checked_fps(row['fps'])
            labels = checked_labels(archive[f'labels_{index}']).copy()
            require(len(labels) == row['frames'] and len(label_spans(labels)) == row['event_count'],
                    'original labels/frame/event metadata differs')
            records.append({**row, 'index': index, 'labels': labels})
    require(len({r['name'] for r in records}) == 77, 'duplicate original row names')
    groups = defaultdict(list)
    for r in records:
        groups[r['sha256']].append(r['index'])
    require(manifest.get('content_groups') == list(groups.values()), 'content group binding differs')
    require(manifest.get('protocol', {}).get('outer_folds') == 5 and
            manifest['protocol'].get('inner_folds') == 3, 'requires fixed outer5/inner3 protocol')
    return records, manifest


def checked_indices(indices, allowed, description):
    require(isinstance(indices, list) and all(type(i) is int for i in indices) and
            len(indices) == len(set(indices)) and set(indices) <= set(allowed),
            'invalid indices: ' + description)
    return set(indices)


def check_partition(records, fit, validation, universe, description):
    a = checked_indices(fit, universe, description + '/fit')
    b = checked_indices(validation, universe, description + '/validation')
    require(a and b and not a & b and a | b == set(universe),
            'not a held-out partition: ' + description)
    require(not {records[i]['sha256'] for i in a} & {records[i]['sha256'] for i in b},
            'same SHA in fit and held-out content: ' + description)


def validate_outer_folds(records, folds, expected_validation=None):
    require(isinstance(folds, list) and folds, 'formal outer folds missing')
    if expected_validation is not None:
        require(len(folds) == len(expected_validation), 'outer fold count differs')
    covered, by_index = [], {}
    for ordinal, fold in enumerate(folds):
        require(type(fold.get('fold')) is int and fold['fold'] == ordinal, 'outer fold ID differs')
        train, validation = fold['train_indices'], fold['validation_indices']
        check_partition(records, train, validation, range(len(records)), f'outer {ordinal}')
        if expected_validation is not None:
            require(validation == expected_validation[ordinal], 'frozen outer partition differs')
        inner_covered = []
        require(isinstance(fold.get('inner_partitions'), list) and fold['inner_partitions'],
                'inner OOF partitions missing')
        for inner_id, part in enumerate(fold['inner_partitions']):
            check_partition(records, part['fit'], part['validation'], train,
                            f'outer {ordinal}/inner {inner_id}')
            inner_covered.extend(part['validation'])
        require(sorted(inner_covered) == sorted(train), 'inner validation coverage differs')
        covered.extend(validation)
        by_index.update({i: ordinal for i in validation})
    require(sorted(covered) == list(range(len(records))), 'outer OOF coverage must be exactly once')
    return by_index


def validate_formal_predictions(records, report, section, fold_prediction_key):
    """Require row-aligned pooled predictions to equal their declared outer fold outputs."""
    require(section in report and isinstance(report[section], dict), 'formal OOF section missing')
    rows = report[section].get('predictions')
    require(isinstance(rows, list) and len(rows) == len(records), 'OOF prediction coverage differs')
    pred = {}
    for index, row in enumerate(rows):
        require(type(row.get('index')) is int and row['index'] == index and
                row.get('name') == records[index]['name'], 'OOF original-row alignment differs')
        pred[index] = checked_intervals(row['segments'], len(records[index]['labels']))
        require(type(row.get('has_candidates')) is bool and
                row['has_candidates'] == bool(pred[index]) and
                type(row.get('is_normal')) is bool and
                row['is_normal'] == (not bool(np.any(records[index]['labels']))),
                'OOF prediction flags differ from original labels')
    for fold in report['folds']:
        values = fold.get(fold_prediction_key)
        require(isinstance(values, dict) and set(values) == {str(i) for i in fold['validation_indices']},
                'fold predictions must contain only outer held-out rows')
        for i in fold['validation_indices']:
            require(checked_intervals(values[str(i)], len(records[i]['labels'])) == pred[i],
                    'pooled predictions differ from formal outer predictions')
    return pred


def verify_fold_artifact(evidence, records, fold, recipe, metadata_path=None, receipt=None):
    if metadata_path is None:
        require(recipe in RECIPE_OWNERS, 'unknown formal training recipe: ' + str(recipe))
        owner = RECIPE_OWNERS[recipe]
        directory = (GROUPED / 'grouped_training/folds' if owner == 'v1'
                     else EXPERIMENTS[owner] / 'training')
        path = directory / f"{fold['fold']}_{recipe}.json"
    else:
        owner, path = 'explicit_report_local', Path(metadata_path)
        require(receipt is not None, 'local cache requires a signed training receipt')
    meta = evidence.json(path)
    require(meta.get('fold', meta.get('scope')) == fold['fold'] and meta.get('recipe') == recipe and
            meta.get('train') == fold['train_indices'] and
            meta.get('validation') == fold['validation_indices'] and
            meta.get('inner_partitions') == fold['inner_partitions'], 'training/OOF partition binding differs')
    require(not meta.get('full_fit') and not meta.get('fullfit') and
            not meta.get('final_selection'), 'fullfit/final-selection cache is not formal outer OOF')
    if owner == 'v1':
        require(meta.get('content_holdout') is True and meta.get('signature') ==
                evidence.hash(GROUPED / 'dataset_manifest.json') +
                json.dumps(meta.get('sources'), sort_keys=True), 'legacy grouped training binding differs')
        fit_proof = 'legacy_content_holdout_metadata_no_separate_fit_evidence'
    else:
        receipt = receipt if metadata_path is not None else evidence.receipt(owner)
        require(meta.get('training_receipt_sha256') == receipt['receipt_sha256'] and
                value_hash({k: v for k, v in meta.items() if k != 'signature'}) == meta.get('signature'),
                'training metadata signature/receipt differs')
        parts = fold['inner_partitions'] + [{'fit': fold['train_indices'],
                                             'validation': fold['validation_indices']}]
        proofs = meta.get('fit_evidence', [])
        require(len(proofs) == len(parts), 'fit-content evidence missing')
        for k, (part, proof) in enumerate(zip(parts, proofs)):
            require(proof.get('partition') == (k if k < len(parts) - 1 else 'outer') and
                    proof.get('fit_content_sha256') == sorted({records[i]['sha256'] for i in part['fit']}),
                    'fit-content proof includes held-out or misses fitted content')
        fit_proof = 'signed_inner_and_outer_fit_content_sha256_verified'
    npz_path = path.with_suffix('.npz')
    data = evidence.read(npz_path, meta.get('npz_sha256'))
    require(valid_digest(meta.get('npz_sha256')), 'probability NPZ hash missing')
    with np.load(BytesIO(data), allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), 'duplicate probability NPZ keys')
        expected = {f'{kind}_{head}_{i}' for kind, ids in (
            ('inner', fold['train_indices']), ('outer', fold['validation_indices']))
                    for i in ids for head in ('frame', 'video')}
        if 'boundary' in recipe:
            expected |= {f'{kind}_boundary_{i}' for kind, ids in (
                ('inner', fold['train_indices']), ('outer', fold['validation_indices'])) for i in ids}
        require(set(archive.files) == expected, 'probability cache is not nested inner/outer OOF')
        for kind, indices in (('inner', fold['train_indices']), ('outer', fold['validation_indices'])):
            for i in indices:
                for head, shape in (('frame', (len(records[i]['labels']),)), ('video', ())):
                    values = archive[f'{kind}_{head}_{i}']
                    require(values.shape == shape and values.dtype.kind in 'biuf' and
                            np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)),
                            'invalid held-out probability frame grid/range')
    return {'recipe': recipe, 'owner': owner, 'metadata_path': path.as_posix(),
            'metadata_sha256': evidence.hash(path), 'probability_path': npz_path.as_posix(),
            'probability_sha256': evidence.hash(npz_path), 'fit_content_verification': fit_proof}


def report_path(version):
    require(version in VERSIONS, 'only formal v1/v4/v5/v8/v9 are allowed')
    return BASELINE_REPORT if version == 'v1' else EXPERIMENTS[version] / 'report.json'


def load_formal_oof(evidence, records, manifest, version):
    path = report_path(version)
    report = evidence.json(path)
    require(report.get('schema_version') == REPORT_SCHEMAS[version] and
            report.get('status') == 'complete', 'not a completed formal nested report: ' + version)
    if version == 'v1':
        require(report.get('dataset_manifest_sha256') == evidence.hash(GROUPED / 'dataset_manifest.json') and
                report.get('base_dataset_npz_sha256') == evidence.hash(ORIGINAL / 'dataset.npz'),
                'grouped v1 original-label binding differs')
        section, fold_key, selected_key = 'grouped_v1_baseline', 'baseline_predictions', 'selected_baseline'
        candidates = {c['name']: [(m['recipe'], m['weight']) for m in c['members']]
                      for c in report['candidates']}
    else:
        receipt = evidence.receipt(version)
        require(report.get('training_receipt_sha256') == receipt['receipt_sha256'] and
                report.get('protocol_sha256') == receipt['protocol_sha256'] and
                ('baseline_report_sha256' not in report or
                 report['baseline_report_sha256'] == receipt['baseline_report_sha256']),
                'formal report receipt/protocol binding differs')
        protocol = evidence.json(EXPERIMENTS[version] / 'protocol.json')
        candidates = dict(protocol['primary_candidates'])
        section, fold_key, selected_key = f'primary_{version}_nested', 'primary_predictions', 'primary'
    by_index = validate_outer_folds(records, report.get('folds'), manifest['outer_folds'])
    require(all(len(f['inner_partitions']) == 3 for f in report['folds']), 'requires three inner OOF folds')
    pred = validate_formal_predictions(records, report, section, fold_key)
    fold_provenance = []
    for fold in report['folds']:
        selected = fold[selected_key]
        require(selected.get('candidate') in candidates, 'selected candidate not in frozen protocol')
        members = candidates[selected['candidate']]
        require(members and all(isinstance(m, (list, tuple)) and len(m) == 2 and
                isinstance(m[1], (int, float)) and math.isfinite(m[1]) and m[1] > 0 for m in members)
                and math.isclose(sum(m[1] for m in members), 1, abs_tol=1e-12),
                'invalid frozen candidate weights')
        artifacts = [verify_fold_artifact(evidence, records, fold, recipe) for recipe, _ in members]
        for artifact in artifacts:
            owner = artifact['owner']
            if version != 'v1' and owner not in ('v1', version) and version in ('v8', 'v9'):
                current, parent = evidence.receipt(version), evidence.receipt(owner)
                require(current.get('control_receipt_sha256') == parent['receipt_sha256'] and
                        all(current.get('control_artifacts', {}).get(artifact[k + '_path']) ==
                            artifact[k + '_sha256'] for k in ('metadata', 'probability')),
                        'selected control artifact lineage differs')
        fold_provenance.append({'fold': fold['fold'], 'train_indices': fold['train_indices'],
            'validation_indices': fold['validation_indices'], 'inner_partitions': fold['inner_partitions'],
            'candidate': selected['candidate'], 'decoder': selected['config'], 'members': members,
            'fit_content_sha256': sorted({records[i]['sha256'] for i in fold['train_indices']}),
            'held_out_content_sha256': sorted({records[i]['sha256'] for i in fold['validation_indices']}),
            'artifacts': artifacts})
    provenance = {'pipeline': version, 'prediction_kind': 'formal_nested_outer_oof',
        'report_path': path.as_posix(), 'report_sha256': evidence.hash(path), 'section': section,
        'fold_prediction_key': fold_key, 'folds': fold_provenance,
        'verification': 'original_label_binding_partitions_SHA_isolation_pooled_outer_identity_cache_hashes_shapes',
        'limitations': 'Structural provenance verified; no decoder/prediction re-execution or model fitting.',
        'fullfit_predictions_used': False}
    return pred, by_index, provenance, report[section]['metrics']



def select_primary_key(report, primary_key=None):
    """Only a nested primary section; ambiguous keys need an explicit choice."""
    pattern = re.compile(r'^primary_[A-Za-z0-9][A-Za-z0-9_]*_nested$')
    choices = [k for k, v in report.items() if pattern.fullmatch(k) and isinstance(v, dict)]
    if primary_key is None:
        require(len(choices) == 1, 'expected one primary_*_nested key; provide --primary-key')
        primary_key = choices[0]
    require(isinstance(primary_key, str) and pattern.fullmatch(primary_key) and primary_key in choices,
            'primary key must name an existing primary_*_nested section, never fullfit/fixed diagnostics')
    require(all(k in report[primary_key] for k in ('metrics', 'predictions')),
            'nested primary section must contain metrics and predictions')
    return primary_key



def direct_original_binding(evidence, receipt, description):
    """Require all original labels/FPS pins; partial mappings are never fallback evidence."""
    pins = receipt.get('input_hashes', {})
    require(isinstance(pins, dict), description + ' invalid input_hashes')
    result = {}
    for path in (ORIGINAL / 'dataset_manifest.json', ORIGINAL / 'dataset.npz',
                 GROUPED / 'dataset_manifest.json'):
        require(path.as_posix() in pins, description + ' missing original-label/FPS binding')
        require(valid_digest(pins[path.as_posix()]), description + ' invalid original-label/FPS hash')
        evidence.read(path, pins[path.as_posix()])
        result[str((evidence.root / path).resolve())] = pins[path.as_posix()]
    return result


def verify_v11_parent_binding(evidence, receipt, receipt_path, schema):
    """Only the frozen proposal-review-v11 parent layout is supported.

    Verify EVERY declared edge, even after a shared node was visited, and every
    parent's direct original-input pins. No hash-based search or inferred paths.
    Canonical receipt hashes and raw-file hashes are intentionally separate.
    """
    require(schema == 'proposal-review-v11', 'transitive receipt schema is not supported')
    require(isinstance(receipt, dict) and valid_digest(receipt.get('receipt_sha256')) and
            value_hash({k: v for k, v in receipt.items() if k != 'receipt_sha256'}) ==
            receipt['receipt_sha256'], 'V11 canonical receipt hash mismatch')
    require(evidence.json(receipt_path) == receipt, 'V11 receipt bytes/body differ')
    paths = {'v4': EXPERIMENTS['v4'] / 'training_receipt.json',
             'v8': EXPERIMENTS['v8'] / 'training_receipt.json',
             'v10': Path('output/algorithm-opt-v10-blocks/training_receipt.json')}
    parents = receipt.get('parent_receipts')
    require(isinstance(parents, dict) and set(parents) == set(paths),
            'V11 requires exactly the supported v4/v8/v10 parent receipts')
    if 'input_hashes' in receipt:
        direct_original_binding(evidence, receipt, 'V11 receipt')
    scalar_edges = {'v8': ('control_receipt_sha256', 'v4'),
                    'v10': ('parent_receipt_sha256', 'v8')}
    nodes, edges, checked, active = [], [], {}, set()

    def visit(alias, expected, child_path):
        require(alias in paths and valid_digest(expected), 'unsupported/invalid parent receipt reference')
        parent_path = paths[alias]
        absolute = str((evidence.root / parent_path).resolve())
        edges.append({'child_receipt_absolute_path': str((evidence.root / child_path).resolve()),
                      'parent_alias': alias, 'parent_receipt_absolute_path': absolute,
                      'expected_canonical_receipt_sha256': expected})
        require(alias not in active, 'cyclic parent receipt chain')
        if alias in checked:
            require(checked[alias] == expected, 'conflicting parent receipt hash reference: ' + alias)
            return
        active.add(alias)
        parent = evidence.json(parent_path)
        require(isinstance(parent, dict) and parent.get('receipt_sha256') == expected and
                value_hash({k: v for k, v in parent.items() if k != 'receipt_sha256'}) == expected,
                'parent canonical receipt hash mismatch: ' + alias)
        pins = direct_original_binding(evidence, parent, 'parent receipt ' + alias)
        references = parent.get('parent_receipts', {})
        require(isinstance(references, dict) and set(references) <= set(paths),
                'unsupported parent receipt chain references: ' + alias)
        references = dict(references)
        for field in ('control_receipt_sha256', 'parent_receipt_sha256'):
            if alias in scalar_edges and scalar_edges[alias][0] == field:
                require(field in parent, 'missing recursive parent receipt reference: ' + alias)
                target = scalar_edges[alias][1]
                require(target not in references or references[target] == parent[field],
                        'conflicting recursive parent receipt references: ' + alias)
                references[target] = parent[field]
            else:
                require(field not in parent, 'unsupported scalar parent receipt reference: ' + alias)
        for target, digest in sorted(references.items()):
            visit(target, digest, parent_path)
        checked[alias] = expected
        active.remove(alias)
        nodes.append({'alias': alias, 'receipt_absolute_path': absolute,
                      'canonical_receipt_sha256': expected,
                      'receipt_file_sha256': evidence.hash(parent_path),
                      'original_label_fps_binding_type': 'direct',
                      'original_label_fps_input_hashes': pins})

    for alias, expected in sorted(parents.items()):
        visit(alias, expected, receipt_path)
    return {'proof_type': 'transitive', 'schema': schema,
            'root_receipt_absolute_path': str((evidence.root / receipt_path).resolve()),
            'root_canonical_receipt_sha256': receipt['receipt_sha256'],
            'root_receipt_file_sha256': evidence.hash(receipt_path),
            'reference_field': 'parent_receipts', 'parent_receipts': nodes, 'parent_edges': edges,
            'verification': 'every_parent_canonical_hash_equals_reference_and_direct_original_labels_FPS_pins'}


def load_explicit_oof(evidence, records, manifest, path, primary_key=None):
    """Future report adapter with strict structural OOF and explicit proof limits.

    Receipt-bound local caches are verified when their selected candidate maps
    directly to a cache. Reused/aliased candidates are NOT guessed: their proof
    availability is recorded, while outer partition/prediction checks still apply.
    """
    absolute = (evidence.root / path).resolve()
    require(absolute.is_relative_to(evidence.root), 'report must be inside the workspace')
    path = absolute.relative_to(evidence.root)
    require(not any(p.startswith(('deployment', 'final_selection', 'product_validation'))
                    for p in path.parts), 'deployment/final-selection outputs are not formal outer OOF')
    report = evidence.json(path)
    require(report.get('status') == 'complete' and not report.get('full_fit') and
            not report.get('fullfit'), 'explicit report is not completed formal nested OOF')
    schema = str(report.get('schema_version', '')).lower()
    require('fullfit' not in schema and 'full-fit' not in schema and 'full_fit' not in schema,
            'fullfit report is not OOF')
    section = select_primary_key(report, primary_key)
    version = section[len('primary_'):-len('_nested')]
    by_index = validate_outer_folds(records, report.get('folds'), manifest['outer_folds'])
    require(all(len(f['inner_partitions']) == manifest.get('protocol', {}).get('inner_folds', 3)
                for f in report['folds']), 'inner OOF count differs from original protocol')
    pred = validate_formal_predictions(records, report, section, 'primary_predictions')
    receipt, receipt_path = None, path.parent / 'training_receipt.json'
    original_binding = {'proof_type': 'unavailable_no_training_receipt'}
    is_v11 = report.get('schema_version') == 'proposal-review-v11'
    require(not is_v11 or valid_digest(report.get('training_receipt_sha256')),
            'proposal-review-v11 requires its signed training receipt')
    if (evidence.root / receipt_path).is_file() or 'training_receipt_sha256' in report:
        receipt = evidence.json(receipt_path)
        require(value_hash({k: v for k, v in receipt.items() if k != 'receipt_sha256'}) ==
                receipt.get('receipt_sha256') and report.get('training_receipt_sha256') ==
                receipt['receipt_sha256'], 'explicit report training receipt binding differs')
        if is_v11:
            original_binding = verify_v11_parent_binding(evidence, receipt, receipt_path,
                                                        report['schema_version'])
        else:
            original_binding = {'proof_type': 'direct',
                'receipt_absolute_path': str((evidence.root / receipt_path).resolve()),
                'canonical_receipt_sha256': receipt['receipt_sha256'],
                'receipt_file_sha256': evidence.hash(receipt_path),
                'original_label_fps_input_hashes': direct_original_binding(
                    evidence, receipt, 'explicit receipt')}
        for key in ('baseline_report_sha256', 'baseline_sha256'):
            if key in receipt:
                evidence.read(BASELINE_REPORT, receipt[key])
        require(report.get('protocol_sha256') == receipt.get('protocol_sha256') and
                valid_digest(receipt.get('protocol_sha256')), 'explicit protocol binding missing/differs')
        evidence.read(path.parent / 'protocol.json', receipt['protocol_sha256'])
    for key, original_path in (('dataset_manifest_sha256', GROUPED / 'dataset_manifest.json'),
                               ('base_dataset_npz_sha256', ORIGINAL / 'dataset.npz')):
        if key in report:
            evidence.read(original_path, report[key])
    folds = []
    for fold in report['folds']:
        selected = fold.get('primary', {})
        require(isinstance(selected.get('candidate'), str) and isinstance(selected.get('config'), dict),
                'formal outer candidate/decoder declaration missing')
        candidate, artifacts = selected['candidate'], []
        require(re.fullmatch(r'[A-Za-z0-9_-]+', candidate), 'unsafe candidate name')
        local_path = path.parent / 'training' / f"{fold['fold']}_{candidate}.json"
        if receipt is not None and (evidence.root / local_path).is_file():
            artifacts.append(verify_fold_artifact(evidence, records, fold, candidate,
                                                  metadata_path=local_path, receipt=receipt))
        elif candidate in RECIPE_OWNERS:
            artifacts.append(verify_fold_artifact(evidence, records, fold, candidate))
        folds.append({'fold': fold['fold'], 'train_indices': fold['train_indices'],
            'validation_indices': fold['validation_indices'], 'inner_partitions': fold['inner_partitions'],
            'candidate': candidate, 'decoder': selected['config'],
            'fit_content_sha256': sorted({records[i]['sha256'] for i in fold['train_indices']}),
            'held_out_content_sha256': sorted({records[i]['sha256'] for i in fold['validation_indices']}),
            'artifacts': artifacts, 'cache_proof_status': ('mapped_selected_cache_verified' if artifacts else
                'no_direct_candidate_cache_mapping_report_partition_proof_only_no_fit_claim')})
    provenance = {'pipeline': version, 'prediction_kind': 'formal_nested_outer_oof',
        'report_path': path.as_posix(), 'report_sha256': evidence.hash(path), 'section': section,
        'fold_prediction_key': 'primary_predictions', 'folds': folds,
        'training_receipt_verified': receipt is not None,
        'original_label_fps_binding': original_binding,
        'verification': 'original_frame_grid_partitions_SHA_isolation_pooled_outer_identity_report_metrics',
        'limitations': 'No decoder/model re-execution. Unmapped aliases have report partition proof only; '
                      'no claim of independently verified model fitting for those folds.',
        'fullfit_predictions_used': False}
    return pred, by_index, provenance, report[section]['metrics']


def aggregate_metrics(videos):
    result = {}
    for threshold in THRESHOLDS:
        key = f'iou_{threshold:.1f}'
        counts = [sum(v['event_matching'][key][k] for v in videos) for k in ('tp', 'fp', 'fn')]
        result[key] = count_metrics(*counts)
    normal_count = sum(v['is_normal'] for v in videos)
    normal_fp = sum(v['normal_false_positive'] for v in videos)
    result['normal'] = {'videos': normal_count, 'false_positive_videos': normal_fp,
                        'false_positive_rate': normal_fp / max(1, normal_count)}
    frame = count_metrics(*(sum(v['frame'][k] for v in videos) for k in ('tp', 'fp', 'fn')))
    result['frame'] = {k: frame[k] for k in ('f1', 'tp', 'fp', 'fn')}
    result['positive_videos_without_candidate'] = sum(v['abnormal_without_candidate'] for v in videos)
    result['predicted_segments'] = sum(v['prediction_count'] for v in videos)
    return result


def verify_report_metrics(actual, expected, label):
    require(isinstance(expected, dict) and set(actual) == set(expected), 'report metric keys differ: ' + label)
    for key, value in actual.items():
        target = expected[key]
        if isinstance(value, dict):
            verify_report_metrics(value, target, label + '/' + key)
        else:
            require(type(target) in (int, float) and math.isfinite(target) and
                    math.isclose(value, target, rel_tol=0, abs_tol=1e-12),
                    'recomputed OOF metric differs: ' + label + '/' + key)


def boundary_summary(events):
    result = {}
    for threshold in THRESHOLDS:
        key = f'iou_{threshold:.1f}'
        matched = [e['boundary_' + key] for e in events if e['boundary_' + key] is not None]
        entry = {'matched_event_count': len(matched), 'conditional_on_matched_events': True}
        for edge in ('start', 'end'):
            offsets = [r[edge + '_offset_seconds'] for r in matched]
            entry[edge] = {'mean_signed_seconds': statistics.mean(offsets) if offsets else None,
                           'median_signed_seconds': statistics.median(offsets) if offsets else None,
                           'mean_absolute_seconds': statistics.mean(abs(x) for x in offsets) if offsets else None}
        result[key] = entry
    return result


def build_error_ledger(root, comparisons=('v1', 'v4', 'v5', 'v8', 'v9'), report=None, primary_key=None):
    require(all(v in VERSIONS for v in comparisons) and len(comparisons) == len(set(comparisons)),
            'invalid/repeated comparison pipeline')
    evidence = InputEvidence(root)
    records, manifest = load_original_records(evidence)
    if report is not None or primary_key is not None:
        primary = load_explicit_oof(evidence, records, manifest,
                                   report if report is not None else report_path('v8'), primary_key)
    else:
        primary = load_formal_oof(evidence, records, manifest, 'v8')
    primary_version = primary[2]['pipeline']
    videos, events, predictions, pipelines, skipped = [], [], [], {}, []
    for version in dict.fromkeys((primary_version,) + tuple(comparisons)):
        if version == primary_version:
            pred, by_index, source, expected = primary
        else:
            if not (evidence.root / report_path(version)).is_file():
                skipped.append({'pipeline': version, 'reason': 'formal_report_missing_no_fullfit_fallback'})
                continue
            pred, by_index, source, expected = load_formal_oof(evidence, records, manifest, version)
        pv, pe, pp = [], [], []
        for record in records:
            fold = source['folds'][by_index[record['index']]]
            row_source = {k: source[k] for k in ('prediction_kind', 'report_path', 'report_sha256', 'section')}
            row_source.update({'fold': fold['fold'], 'candidate': fold['candidate']})
            v, e, p = analyze_video(record, pred[record['index']], version, row_source)
            pv.append(v)
            pe.extend(e)
            pp.extend(p)
        metric_groups = {'all': aggregate_metrics(pv),
            'normal': aggregate_metrics([v for v in pv if v['event_count'] == 0]),
            'single_event': aggregate_metrics([v for v in pv if v['event_count'] == 1]),
            'multi_event': aggregate_metrics([v for v in pv if v['event_count'] > 1])}
        verify_report_metrics(metric_groups, expected, version)
        pipelines[version] = {'metrics': metric_groups, 'oof_provenance': source,
            'report_metrics_reproduced': True, 'video_rows': len(pv), 'event_rows': len(pe),
            'prediction_rows': len(pp), 'error_video_counts': dict(Counter(
                flag for v in pv for flag in v['error_types'])),
            'split_proxy_events': sum(e['split_proxy'] for e in pe),
            'merge_proxy_predictions': sum(p['merge_proxy'] for p in pp),
            'normal_false_positive_segments': sum(v['prediction_count'] for v in pv if v['is_normal']),
            'boundary_offsets': boundary_summary(pe)}
        videos.extend(pv)
        events.extend(pe)
        predictions.extend(pp)
    aliases = alias_conflicts(records)
    summary = {'schema_version': SCHEMA, 'status': 'complete', 'primary_pipeline': primary_version,
        'role': 'previously_inspected_development_OOF_not_new_blind_test',
        'dataset': {'rows': len(records), 'unique_contents': len({r['sha256'] for r in records}),
                    'frames': sum(len(r['labels']) for r in records),
                    'events': sum(len(label_spans(r['labels'])) for r in records),
                    'normal_rows': sum(not np.any(r['labels']) for r in records),
                    'alias_groups': len(aliases), 'label_conflict_groups': sum(
                        a['label_conflict'] is True for a in aliases)},
        'definitions': {'frame_intervals': 'zero_based_inclusive_[start,end]',
            'time_intervals': 'half_open_[start/fps,(end+1)/fps); original manifest FPS only',
            'boundary_offsets': 'prediction_minus_GT; positive=start late/end overhang; matched events only',
            'event_matching': 'prediction_order_greedy; first GT on ties; >= threshold; one-to-one separately at .3/.5',
            'split_merge': 'positive_frame_overlap_graph_degree>1; geometric proxies, not video semantic causes',
            'counting': 'original annotation rows preserved, not deduplicated; aliases kept in same fold',
            'semantic_cause': 'unknown for all rows; no attribution to camera/lighting/motion/model channel'},
        'pipelines': {v: {k: x for k, x in p.items() if k != 'oof_provenance'} for v, p in pipelines.items()},
        'skipped_comparisons': skipped}
    provenance = {'schema_version': SCHEMA, 'created_at': datetime.now(timezone.utc).isoformat(),
        'workspace_root': str(evidence.root), 'input_hashes': dict(sorted(evidence.input_hashes.items())),
        'original_labels': {'manifest': (ORIGINAL / 'dataset_manifest.json').as_posix(),
                            'archive': (ORIGINAL / 'dataset.npz').as_posix(), 'keys': 'labels_0..labels_76'},
        'video_content_hashes': 'Taken from frozen original manifest; video bytes not rehashed/decoded.',
        'oof_pipelines': {v: p['oof_provenance'] for v, p in pipelines.items()},
        'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'training_performed': False, 'inference_performed': False, 'remote_services_called': False,
        'labels_changed': False, 'fullfit_predictions_used': False}
    return {'summary': summary, 'videos': videos, 'events': events, 'predictions': predictions,
            'alias_conflicts': aliases, 'provenance': provenance}


def flattened_row(row, prefix=''):
    result = {}
    for key, value in row.items():
        name = prefix + str(key)
        if isinstance(value, dict):
            result.update(flattened_row(value, name + '.'))
        elif isinstance(value, (list, tuple)):
            result[name] = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        else:
            result[name] = value
    return result


def write_ledger(output, ledger):
    """Claim a brand-new directory and files exclusively; completion marker is last.

    Validation happens before this function. On an I/O failure a partial new
    directory is retained, without completion.json; existing data is never deleted.
    """
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError('output already exists; choose a new path: ' + str(output))
    output.mkdir(parents=True, exist_ok=False)
    written = {}
    for name in ('summary', 'videos', 'events', 'predictions', 'alias_conflicts', 'provenance'):
        path = output / (name + '.json')
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(ledger[name], handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        written[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in ('videos', 'events', 'predictions', 'alias_conflicts'):
            rows = [flattened_row(r) for r in ledger[name]]
            fields = list(dict.fromkeys(k for r in rows for k in r))
            if not fields:
                fields = ['pipeline', 'index', 'semantic_cause'] if name != 'alias_conflicts' else [
                    'sha256', 'indices', 'label_conflict', 'conflicting_frame_indices']
            path = output / (name + '.csv')
            with path.open('x', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    # Keep filenames/strings from becoming formulas when opened in Excel.
                    writer.writerow({k: ("'" + v if isinstance(v, str) and v[:1] in ('=', '+', '-', '@') else v)
                                     for k, v in row.items()})
            written[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (output / 'completion.json').open('x', encoding='utf-8', newline='\n') as handle:
        json.dump({'schema_version': SCHEMA, 'status': 'complete', 'output_hashes': written},
                  handle, indent=2, allow_nan=False)
        handle.write('\n')
    return output
