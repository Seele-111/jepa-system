"""Focused unit tests: geometry, original timing, provenance and exclusive output.

Run with: python -B -m unittest discover -s code -p test_optimized_error_ledger.py -v
All synthetic assets/output live in TemporaryDirectory, never frozen folders.
"""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from build_optimization_error_ledger import check_output_path
from optimized_error_ledger import (BASELINE_REPORT, EXPERIMENTS, GROUPED, ORIGINAL,
    InputEvidence, LedgerError, aggregate_metrics, alias_conflicts, analyze_video, load_explicit_oof,
    boundary_offsets, boundary_summary, checked_labels, label_spans, match_events,
    report_path, select_primary_key, timed_interval, validate_formal_predictions, validate_outer_folds,
    value_hash, verify_fold_artifact, verify_report_metrics, write_ledger)


def record(index, labels, fps=2., sha=None):
    return {'index': index, 'name': f'视频_{index}.mp4', 'fps': fps,
            'frames': len(labels), 'labels': np.asarray(labels, dtype=np.int8),
            'sha256': sha or hashlib.sha256(str(index).encode()).hexdigest()}


def four_records():
    return [record(i, [0, 1, 1, 0, 0, 0]) for i in range(4)]


def two_folds():
    folds = []
    for fold, val in enumerate(([0, 1], [2, 3])):
        train = [i for i in range(4) if i not in val]
        folds.append({'fold': fold, 'train_indices': train, 'validation_indices': val,
            'inner_partitions': [{'fit': [train[1]], 'validation': [train[0]], 'seed': 101},
                                 {'fit': [train[0]], 'validation': [train[1]], 'seed': 102}],
            'primary_predictions': {str(i): [[1, 2]] for i in val}})
    return folds


def formal_report(records):
    return {'folds': two_folds(), 'primary_v8_nested': {'predictions': [
        {'index': r['index'], 'name': r['name'], 'segments': [[1, 2]],
         'has_candidates': True, 'is_normal': False} for r in records]}}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def synthetic_cache(root, owner='v1'):
    """Create an actual nested probability archive, not a mock model or service."""
    records, fold = four_records(), two_folds()[0]
    grouped = root / GROUPED / 'dataset_manifest.json'
    write_json(grouped, {'fixture': True})
    grouped_hash = hashlib.sha256(grouped.read_bytes()).hexdigest()
    recipe = 'corrected_motion_rf' if owner == 'v1' else 'event_compact_rgb_corrected_motion_et'
    directory = GROUPED / 'grouped_training/folds' if owner == 'v1' else EXPERIMENTS['v8'] / 'training'
    path = root / directory / ('0_' + recipe + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f'{kind}_{head}_{i}': (np.full(6, .4) if head == 'frame' else np.asarray(.8))
              for kind, ids in (('inner', fold['train_indices']), ('outer', fold['validation_indices']))
              for i in ids for head in ('frame', 'video')}
    np.savez(path.with_suffix('.npz'), **arrays)
    meta = {'fold': 0, 'recipe': recipe, 'train': fold['train_indices'],
            'validation': fold['validation_indices'], 'inner_partitions': fold['inner_partitions'],
            'npz_sha256': hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()}
    if owner == 'v1':
        meta.update({'content_holdout': True, 'sources': {}, 'signature': grouped_hash + '{}'})
    else:
        for p in (ORIGINAL / 'dataset_manifest.json', BASELINE_REPORT, EXPERIMENTS['v8'] / 'protocol.json'):
            write_json(root / p, {'fixture': True})
        (root / ORIGINAL / 'dataset.npz').write_bytes(b'unit-original-label-binding')
        receipt = {'input_hashes': {p.as_posix(): hashlib.sha256((root / p).read_bytes()).hexdigest()
                   for p in (ORIGINAL / 'dataset_manifest.json', ORIGINAL / 'dataset.npz',
                             GROUPED / 'dataset_manifest.json')},
            'baseline_report_sha256': hashlib.sha256((root / BASELINE_REPORT).read_bytes()).hexdigest(),
            'protocol_sha256': hashlib.sha256((root / EXPERIMENTS['v8'] / 'protocol.json').read_bytes()).hexdigest()}
        receipt['receipt_sha256'] = value_hash(receipt)
        write_json(root / EXPERIMENTS['v8'] / 'training_receipt.json', receipt)
        parts = fold['inner_partitions'] + [{'fit': fold['train_indices']}]
        meta['fit_evidence'] = [{'partition': k if k < len(parts) - 1 else 'outer',
            'fit_content_sha256': sorted({records[i]['sha256'] for i in p['fit']})}
                               for k, p in enumerate(parts)]
        meta['training_receipt_sha256'] = receipt['receipt_sha256']
        meta['signature'] = value_hash(meta)
    write_json(path, meta)
    return records, fold, recipe, path, meta


def v11_parent_paths():
    return {'v4': EXPERIMENTS['v4'] / 'training_receipt.json',
            'v8': EXPERIMENTS['v8'] / 'training_receipt.json',
            'v10': Path('output/algorithm-opt-v10-blocks/training_receipt.json')}


def save_v11_chain(root, path, parents, receipt, report, rewrite_edges=True):
    """Sign synthetic ancestors and references after a controlled mutation."""
    for alias in ('v4', 'v8', 'v10'):
        if rewrite_edges and alias == 'v8':
            parents[alias]['control_receipt_sha256'] = parents['v4']['receipt_sha256']
        if rewrite_edges and alias == 'v10':
            parents[alias]['parent_receipt_sha256'] = parents['v8']['receipt_sha256']
        parents[alias]['receipt_sha256'] = value_hash({
            k: v for k, v in parents[alias].items() if k != 'receipt_sha256'})
        write_json(root / v11_parent_paths()[alias], parents[alias])
    receipt['parent_receipts'] = {k: p['receipt_sha256'] for k, p in parents.items()}
    receipt['receipt_sha256'] = value_hash({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    write_json(root / path.parent / 'training_receipt.json', receipt)
    report['training_receipt_sha256'] = receipt['receipt_sha256']
    write_json(root / path, report)


def v11_receipt_fixture(root):
    path = Path('output/unit-v11/report.json')
    for p in (ORIGINAL / 'dataset_manifest.json', GROUPED / 'dataset_manifest.json',
              BASELINE_REPORT, path.parent / 'protocol.json'):
        write_json(root / p, {'fixture': True})
    (root / ORIGINAL / 'dataset.npz').write_bytes(b'unit-original-label-binding')
    pins = {p.as_posix(): hashlib.sha256((root / p).read_bytes()).hexdigest() for p in (
        ORIGINAL / 'dataset_manifest.json', ORIGINAL / 'dataset.npz', GROUPED / 'dataset_manifest.json')}
    parents = {alias: {'input_hashes': deepcopy(pins)} for alias in v11_parent_paths()}
    receipt = {'protocol_sha256': hashlib.sha256((root / path.parent / 'protocol.json').read_bytes()).hexdigest(),
               'baseline_sha256': hashlib.sha256((root / BASELINE_REPORT).read_bytes()).hexdigest()}
    records = four_records()
    report = formal_report(records)
    report['primary_v11_nested'] = report.pop('primary_v8_nested')
    report['primary_v11_nested']['metrics'] = {}
    report.update({'schema_version': 'proposal-review-v11', 'status': 'complete',
                   'protocol_sha256': receipt['protocol_sha256']})
    for fold in report['folds']:
        fold['primary'] = {'candidate': 'unit_unmapped_v11', 'config': {'threshold': .5}}
    save_v11_chain(root, path, parents, receipt, report)
    manifest = {'outer_folds': [[0, 1], [2, 3]], 'protocol': {'inner_folds': 2}}
    return records, manifest, path, parents, receipt, report


class MatchingAndTimingTests(unittest.TestCase):
    def test_inclusive_single_frame_and_exact_thresholds(self):
        self.assertEqual(match_events([(2, 2)], [(2, 2)], .5)['tp'], 1)
        self.assertEqual(match_events([(0, 9)], [(0, 2)], .3)['tp'], 1)
        self.assertEqual(match_events([(0, 9)], [(0, 4)], .5)['tp'], 1)
        self.assertEqual(match_events([(0, 9)], [(0, 2)], .5)['tp'], 0)

    def test_prediction_order_and_first_gt_tie_are_preserved(self):
        truth = [(0, 3), (6, 9)]
        forward = match_events([(0, 9), (0, 3)], truth, .3)
        reverse = match_events([(0, 3), (0, 9)], truth, .3)
        self.assertEqual(forward['matches'][0]['event_id'], 0)
        self.assertEqual(forward['tp'], 1)
        self.assertEqual(reverse['tp'], 2)  # No hidden Hungarian/sorting replacement.

    def test_duplicate_predictions_match_only_once_but_frames_union(self):
        v, e, p = analyze_video(record(0, [0, 1, 1, 0]), [(1, 2), (1, 2)])
        self.assertEqual(v['event_matching']['iou_0.5']['tp'], 1)
        self.assertEqual(v['event_matching']['iou_0.5']['fp'], 1)
        self.assertEqual(v['frame']['tp'], 2)
        self.assertEqual(v['frame']['fp'], 0)
        self.assertTrue(e[0]['split_proxy'])
        self.assertEqual(len(p), 2)

    def test_zero_overlap_and_empty_matching(self):
        result = match_events([(1, 1)], [(2, 2)], .3)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 1, 1))
        self.assertEqual(match_events([], [], .5)['matches'], [])
        self.assertEqual(match_events([], [(0, 1)], .5)['unmatched_event_ids'], [0])

    def test_original_fps_half_open_time_and_no_rounding(self):
        interval = timed_interval((24, 47), 24.)
        self.assertEqual(interval['start_seconds'], 1.)
        self.assertEqual(interval['end_seconds_exclusive'], 2.)
        self.assertEqual(interval['duration_seconds'], 1.)
        self.assertAlmostEqual(timed_interval((0, 0), 29.97)['end_seconds_exclusive'], 1 / 29.97)
        self.assertEqual(boundary_offsets((4, 9), (2, 7), 2.)['start_offset_seconds'], 1.)
        self.assertEqual(boundary_offsets((1, 6), (2, 7), 2.)['end_offset_seconds'], -.5)

    def test_invalid_labels_fps_intervals_and_threshold_rejected(self):
        for labels in ([], [0, 2], [0, float('nan')], [[0, 1]]):
            with self.subTest(labels=labels), self.assertRaises(LedgerError):
                checked_labels(labels)
        for fps in (0, -1, float('inf'), float('nan'), True):
            with self.subTest(fps=fps), self.assertRaises(LedgerError):
                timed_interval((0, 0), fps)
        for span in ((-1, 0), (2, 1), (1.5, 2), (True, 2)):
            with self.subTest(span=span), self.assertRaises(LedgerError):
                timed_interval(span, 24.)
        with self.assertRaises(LedgerError):
            analyze_video(record(0, [0, 0]), [(0, 2)])
        with self.assertRaises(LedgerError):
            match_events([], [], 0)


class ErrorDecompositionTests(unittest.TestCase):
    def test_original_event_extraction_and_normal_false_positive(self):
        self.assertEqual(label_spans([1, 0, 1, 1, 0, 1]), [(0, 0), (2, 3), (5, 5)])
        v, events, pred = analyze_video(record(0, [0, 0, 0, 0]), [(0, 1)])
        self.assertTrue(v['normal_false_positive'])
        self.assertEqual(events, [])
        self.assertEqual(v['event_matching']['iou_0.3']['fp'], 1)
        self.assertEqual(v['frame']['fp'], 2)
        self.assertIn('normal_false_positive', pred[0]['error_types'])
        clean, _, _ = analyze_video(record(1, [0, 0]), [])
        self.assertEqual(clean['error_types'], [])

    def test_abnormal_without_candidate_has_no_invented_boundaries(self):
        v, events, pred = analyze_video(record(0, [1, 1, 0, 1]), [])
        self.assertTrue(v['abnormal_without_candidate'])
        self.assertEqual(v['event_matching']['iou_0.5']['fn'], 2)
        self.assertEqual(pred, [])
        self.assertTrue(all(e['boundary_iou_0.3'] is None for e in events))
        self.assertEqual(boundary_summary(events)['iou_0.5']['matched_event_count'], 0)
        self.assertIsNone(boundary_summary(events)['iou_0.5']['start']['mean_absolute_seconds'])

    def test_split_proxy_does_not_count_an_event_twice(self):
        v, events, _ = analyze_video(record(0, [1, 1, 1, 1, 1, 1]), [(0, 1), (4, 5)])
        self.assertEqual(v['split_proxy_event_ids'], [0])
        self.assertTrue(events[0]['diagnostic_only'])
        self.assertEqual(v['event_matching']['iou_0.3']['tp'], 1)
        self.assertEqual(v['event_matching']['iou_0.5']['tp'], 0)

    def test_merge_proxy_does_not_turn_best_overlap_into_a_match(self):
        v, events, pred = analyze_video(record(0, [1, 1, 0, 1, 1]), [(0, 4)])
        self.assertEqual(v['merge_proxy_prediction_ids'], [0])
        self.assertTrue(pred[0]['merge_proxy'])
        self.assertEqual(v['event_matching']['iou_0.3']['tp'], 1)
        self.assertIsNone(events[1]['match_iou_0.3'])
        self.assertTrue(events[1]['best_overlap_diagnostic']['not_an_additional_match'])

    def test_boundary_deltas_are_signed_and_only_matched_in_summary(self):
        _, events, _ = analyze_video(record(0, [0, 0, 1, 1, 1, 1, 1, 1, 0]), [(3, 8)])
        self.assertEqual(events[0]['boundary_iou_0.5']['start_offset_frames'], 1)
        self.assertEqual(events[0]['boundary_iou_0.5']['end_offset_seconds'], .5)
        summary = boundary_summary(events)['iou_0.5']
        self.assertTrue(summary['conditional_on_matched_events'])
        self.assertEqual(summary['start']['mean_absolute_seconds'], .5)

    def test_aggregate_normal_empty_and_report_metric_guard(self):
        videos = [analyze_video(record(0, [0, 0]), [(0, 0)])[0],
                  analyze_video(record(1, [1, 1]), [])[0]]
        result = aggregate_metrics(videos)
        self.assertEqual(result['normal']['false_positive_videos'], 1)
        self.assertEqual(result['positive_videos_without_candidate'], 1)
        self.assertEqual(result['iou_0.5']['fn'], 1)
        verify_report_metrics(result, deepcopy(result), 'synthetic')
        changed = deepcopy(result)
        changed['frame']['fp'] += 1
        with self.assertRaises(LedgerError):
            verify_report_metrics(result, changed, 'synthetic')

    def test_all_semantic_causes_stay_unknown_and_input_labels_unchanged(self):
        r = record(0, [1, 1, 0, 1, 1])
        before = r['labels'].copy()
        v, events, pred = analyze_video(r, [(0, 4)])
        self.assertTrue(all(x['semantic_cause'] == 'unknown' for x in [v] + events + pred))
        np.testing.assert_array_equal(r['labels'], before)

    def test_same_sha_alias_conflicts_preserve_both_annotations(self):
        rows = [record(0, [0, 1, 0, 1]), record(1, [0, 0, 0, 0], sha='b' * 64),
                record(2, [0, 0, 0, 0], sha=hashlib.sha256(b'0').hexdigest())]
        before = [r['labels'].copy() for r in rows]
        conflict = alias_conflicts(rows)[0]
        self.assertEqual(conflict['indices'], [0, 2])
        self.assertEqual(conflict['conflicting_frame_indices'], [1, 3])
        self.assertTrue(conflict['label_conflict'])
        self.assertEqual(conflict['conflicting_frames'][0]['original_labels_by_index'], {'0': 1, '2': 0})
        for r, y in zip(rows, before):
            np.testing.assert_array_equal(r['labels'], y)

    def test_alias_metadata_mismatch_is_not_guessed_or_retimed(self):
        rows = [record(0, [0, 1]), record(1, [0, 1], fps=4., sha=hashlib.sha256(b'0').hexdigest())]
        entry = alias_conflicts(rows)[0]
        self.assertFalse(entry['metadata_comparable'])
        self.assertIsNone(entry['label_conflict'])
        self.assertIsNone(entry['conflicting_frame_indices'])


class OOFProvenanceTests(unittest.TestCase):
    def test_valid_outer_and_inner_partitions(self):
        records, folds = four_records(), two_folds()
        self.assertEqual(validate_outer_folds(records, folds, [[0, 1], [2, 3]]),
                         {0: 0, 1: 0, 2: 1, 3: 1})
        report = formal_report(records)
        self.assertEqual(validate_formal_predictions(records, report, 'primary_v8_nested',
                                                     'primary_predictions')[0], [(1, 2)])

    def test_alias_across_fit_validation_is_rejected(self):
        records = four_records()
        records[2]['sha256'] = records[0]['sha256']
        with self.assertRaisesRegex(LedgerError, 'same SHA'):
            validate_outer_folds(records, two_folds())

    def test_fullfit_partition_inner_leakage_and_duplicate_outer_coverage_rejected(self):
        for mode in ('fullfit', 'inner_leakage', 'duplicate_outer'):
            folds = two_folds()
            if mode == 'fullfit':
                folds[0]['train_indices'] = [0, 1, 2, 3]
            elif mode == 'inner_leakage':
                folds[0]['inner_partitions'][0]['fit'] = [0, 3]
            else:
                folds[1] = deepcopy(folds[0])
                folds[1]['fold'] = 1
            with self.subTest(mode=mode), self.assertRaises(LedgerError):
                validate_outer_folds(four_records(), folds)

    def test_pooled_tamper_extra_fit_row_and_fullfit_only_section_rejected(self):
        records = four_records()
        for mode in ('tamper', 'fit_row', 'fullfit_only'):
            report = formal_report(records)
            if mode == 'tamper':
                report['primary_v8_nested']['predictions'][0]['segments'] = [[0, 2]]
            elif mode == 'fit_row':
                report['folds'][0]['primary_predictions']['2'] = [[1, 2]]
            else:
                report['deployment_fullfit'] = report.pop('primary_v8_nested')
            with self.subTest(mode=mode), self.assertRaises(LedgerError):
                validate_formal_predictions(records, report, 'primary_v8_nested', 'primary_predictions')
        with self.assertRaises(LedgerError):
            report_path('fullfit')

    def test_legacy_nested_cache_hashes_and_outer_shapes_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, fold, recipe, _, _ = synthetic_cache(root)
            evidence = InputEvidence(root)
            proof = verify_fold_artifact(evidence, records, fold, recipe)
            self.assertEqual(proof['owner'], 'v1')
            self.assertEqual(proof['fit_content_verification'],
                             'legacy_content_holdout_metadata_no_separate_fit_evidence')
            self.assertEqual(len(evidence.input_hashes), 3)

    def test_cache_fullfit_flag_wrong_hash_and_wrong_outer_grid_rejected(self):
        for mode in ('fullfit', 'wrong_hash', 'wrong_grid'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                records, fold, recipe, path, meta = synthetic_cache(root)
                if mode == 'fullfit':
                    meta['full_fit'] = True
                elif mode == 'wrong_hash':
                    meta['npz_sha256'] = '0' * 64
                else:
                    with np.load(path.with_suffix('.npz'), allow_pickle=False) as a:
                        arrays = {k: a[k].copy() for k in a.files}
                    arrays['outer_frame_0'] = np.zeros(5)
                    np.savez(path.with_suffix('.npz'), **arrays)
                    meta['npz_sha256'] = hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()
                write_json(path, meta)
                with self.assertRaises(LedgerError):
                    verify_fold_artifact(InputEvidence(root), records, fold, recipe)

    def test_new_signed_fit_proof_and_heldout_content_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, fold, recipe, path, meta = synthetic_cache(root, owner='v8')
            proof = verify_fold_artifact(InputEvidence(root), records, fold, recipe)
            self.assertEqual(proof['fit_content_verification'], 'signed_inner_and_outer_fit_content_sha256_verified')
            meta['fit_evidence'][-1]['fit_content_sha256'].append(records[0]['sha256'])
            meta['signature'] = value_hash({k: v for k, v in meta.items() if k != 'signature'})
            write_json(path, meta)
            with self.assertRaisesRegex(LedgerError, 'fit-content proof'):
                verify_fold_artifact(InputEvidence(root), records, fold, recipe)

    def test_input_hash_recording_tamper_and_workspace_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'input.json'
            write_json(path, {'original': True})
            evidence = InputEvidence(root)
            evidence.json('input.json')
            self.assertEqual(evidence.input_hashes['input.json'], hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(LedgerError):
                InputEvidence(root).read('input.json', '0' * 64)
            write_json(path, {'original': False})
            with self.assertRaisesRegex(LedgerError, 'input changed'):
                evidence.json('input.json')
            with self.assertRaisesRegex(LedgerError, 'escapes workspace'):
                evidence.read('../outside.json')


class GenericReportTests(unittest.TestCase):
    def section(self):
        return {'metrics': {}, 'predictions': [], 'summary': {}}

    def test_unique_future_primary_key_auto_detection(self):
        for key in ('primary_v10_nested', 'primary_v11_nested', 'primary_future_blocks_nested'):
            with self.subTest(key=key):
                report = {key: self.section(), 'deployment_fullfit': {'predictions': []}}
                self.assertEqual(select_primary_key(report), key)
                self.assertEqual(select_primary_key(report, key), key)

    def test_ambiguous_primary_keys_require_explicit_not_best_score(self):
        report = {'primary_v10_nested': self.section(), 'primary_v11_nested': self.section()}
        with self.assertRaisesRegex(LedgerError, '--primary-key'):
            select_primary_key(report)
        self.assertEqual(select_primary_key(report, 'primary_v10_nested'), 'primary_v10_nested')

    def test_explicit_fullfit_fixed_missing_and_malformed_keys_rejected(self):
        report = {'primary_v11_nested': self.section(), 'deployment_fullfit': self.section(),
                  'fixed_candidate_diagnostics': self.section()}
        for key in ('deployment_fullfit', 'fixed_candidate_diagnostics', 'primary_v10_nested',
                    'primary_../escape_nested'):
            with self.subTest(key=key), self.assertRaises(LedgerError):
                select_primary_key(report, key)
        with self.assertRaises(LedgerError):
            select_primary_key({'primary_v11_nested': {'predictions': []}})

    def test_future_outer_report_adapter_records_unmapped_proof_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, report = four_records(), formal_report(four_records())
            report['primary_v11_nested'] = report.pop('primary_v8_nested')
            report['primary_v11_nested']['metrics'] = {}
            report.update({'schema_version': 'future-test', 'status': 'complete'})
            for fold in report['folds']:
                fold['primary'] = {'candidate': 'new_unmapped_alias', 'config': {'threshold': .5}}
            path = Path('output/future-experiment/report.json')
            write_json(root / path, report)
            manifest = {'outer_folds': [[0, 1], [2, 3]], 'protocol': {'inner_folds': 2}}
            _, _, provenance, _ = load_explicit_oof(InputEvidence(root), records, manifest, path)
            self.assertEqual(provenance['pipeline'], 'v11')
            self.assertFalse(provenance['training_receipt_verified'])
            self.assertFalse(provenance['fullfit_predictions_used'])
            self.assertIn('no_fit_claim', provenance['folds'][0]['cache_proof_status'])
            report['full_fit'] = True
            write_json(root / path, report)
            with self.assertRaises(LedgerError):
                load_explicit_oof(InputEvidence(root), records, manifest, path)

    def test_future_scope_cache_layout_is_verified_not_final_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, fold, recipe, path, meta = synthetic_cache(root, owner='v8')
            meta['scope'] = meta.pop('fold')
            meta['final_selection'] = False
            meta['signature'] = value_hash({k: v for k, v in meta.items() if k != 'signature'})
            write_json(path, meta)
            receipt = json.loads((root / EXPERIMENTS['v8'] / 'training_receipt.json').read_text(encoding='utf-8'))
            proof = verify_fold_artifact(InputEvidence(root), records, fold, recipe,
                                        metadata_path=path.relative_to(root), receipt=receipt)
            self.assertEqual(proof['owner'], 'explicit_report_local')
            meta['final_selection'] = True
            meta['signature'] = value_hash({k: v for k, v in meta.items() if k != 'signature'})
            write_json(path, meta)
            with self.assertRaisesRegex(LedgerError, 'final-selection'):
                verify_fold_artifact(InputEvidence(root), records, fold, recipe,
                                     metadata_path=path.relative_to(root), receipt=receipt)

    def test_custom_frozen_report_folder_cannot_be_output_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / 'output/future-experiment/report.json'
            with self.assertRaises(LedgerError):
                check_output_path(root, report.parent / 'ledger', report)


class V11ParentReceiptTests(unittest.TestCase):
    def test_valid_recursive_chain_is_transitive_with_absolute_parent_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, manifest, path, parents, _, _ = v11_receipt_fixture(root)
            evidence = InputEvidence(root)
            _, _, provenance, _ = load_explicit_oof(evidence, records, manifest, path)
            binding = provenance['original_label_fps_binding']
            self.assertEqual(binding['proof_type'], 'transitive')
            self.assertTrue(provenance['training_receipt_verified'])
            self.assertEqual(len(binding['parent_edges']), 5)  # root three + v10->v8 + v8->v4.
            self.assertEqual({p['alias'] for p in binding['parent_receipts']}, {'v4', 'v8', 'v10'})
            for parent in binding['parent_receipts']:
                alias = parent['alias']
                source = root / v11_parent_paths()[alias]
                self.assertEqual(Path(parent['receipt_absolute_path']), source.resolve())
                self.assertEqual(parent['canonical_receipt_sha256'], parents[alias]['receipt_sha256'])
                self.assertEqual(parent['receipt_file_sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
                self.assertNotEqual(parent['canonical_receipt_sha256'], parent['receipt_file_sha256'])
                self.assertEqual(parent['original_label_fps_binding_type'], 'direct')
                self.assertEqual(len(parent['original_label_fps_input_hashes']), 3)
                self.assertTrue(all(Path(p).is_absolute() for p in parent['original_label_fps_input_hashes']))

    def test_tampered_parent_body_without_resigning_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, manifest, path, parents, _, _ = v11_receipt_fixture(root)
            parents['v8']['tampered'] = True
            write_json(root / v11_parent_paths()['v8'], parents['v8'])
            with self.assertRaisesRegex(LedgerError, 'parent canonical receipt hash mismatch'):
                load_explicit_oof(InputEvidence(root), records, manifest, path)

    def test_wrong_root_parent_reference_rejected_even_when_root_is_resigned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, manifest, path, _, receipt, report = v11_receipt_fixture(root)
            receipt['parent_receipts']['v8'] = '0' * 64
            receipt['receipt_sha256'] = value_hash({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
            write_json(root / path.parent / 'training_receipt.json', receipt)
            report['training_receipt_sha256'] = receipt['receipt_sha256']
            write_json(root / path, report)
            with self.assertRaisesRegex(LedgerError, 'parent receipt hash reference'):
                load_explicit_oof(InputEvidence(root), records, manifest, path)

    def test_wrong_recursive_parent_hash_rejected_with_valid_child_canonical_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, manifest, path, parents, receipt, report = v11_receipt_fixture(root)
            parents['v10']['parent_receipt_sha256'] = '0' * 64
            save_v11_chain(root, path, parents, receipt, report, rewrite_edges=False)
            with self.assertRaisesRegex(LedgerError, 'parent canonical receipt hash mismatch'):
                load_explicit_oof(InputEvidence(root), records, manifest, path)

    def test_missing_labels_or_fps_pins_rejected_even_when_entire_chain_is_resigned(self):
        for missing in (ORIGINAL / 'dataset.npz', ORIGINAL / 'dataset_manifest.json',
                        GROUPED / 'dataset_manifest.json'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                records, manifest, path, parents, receipt, report = v11_receipt_fixture(root)
                del parents['v8']['input_hashes'][missing.as_posix()]
                save_v11_chain(root, path, parents, receipt, report)
                with self.assertRaisesRegex(LedgerError, 'missing original-label/FPS binding'):
                    load_explicit_oof(InputEvidence(root), records, manifest, path)

    def test_wrong_original_input_hash_rejected_after_resigning_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, manifest, path, parents, receipt, report = v11_receipt_fixture(root)
            parents['v4']['input_hashes'][(ORIGINAL / 'dataset.npz').as_posix()] = '0' * 64
            save_v11_chain(root, path, parents, receipt, report)
            with self.assertRaisesRegex(LedgerError, 'input hash mismatch'):
                load_explicit_oof(InputEvidence(root), records, manifest, path)

    def test_null_or_malformed_original_pins_are_not_treated_as_optional_hashes(self):
        for invalid in (None, True, 'not-a-hash', 'G' * 64):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                records, manifest, path, parents, receipt, report = v11_receipt_fixture(root)
                parents['v8']['input_hashes'][(ORIGINAL / 'dataset.npz').as_posix()] = invalid
                save_v11_chain(root, path, parents, receipt, report)
                with self.assertRaisesRegex(LedgerError, 'invalid original-label/FPS hash'):
                    load_explicit_oof(InputEvidence(root), records, manifest, path)

    def test_missing_parent_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, manifest, path, _, _, _ = v11_receipt_fixture(root)
            (root / v11_parent_paths()['v4']).unlink()
            with self.assertRaises(FileNotFoundError):
                load_explicit_oof(InputEvidence(root), records, manifest, path)

    def test_unsupported_schema_does_not_get_transitive_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, manifest, path, _, _, report = v11_receipt_fixture(root)
            report['schema_version'] = 'unknown-review-v12'
            write_json(root / path, report)
            with self.assertRaisesRegex(LedgerError, 'explicit receipt missing original-label/FPS binding'):
                load_explicit_oof(InputEvidence(root), records, manifest, path)

    def test_cycles_unknown_aliases_and_missing_parent_references_fail_closed(self):
        for mode in ('cycle', 'unknown_alias', 'missing_recursive'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                records, manifest, path, parents, receipt, report = v11_receipt_fixture(root)
                if mode == 'cycle':
                    parents['v4']['parent_receipts'] = {'v4': '0' * 64}
                elif mode == 'unknown_alias':
                    parents['v4']['parent_receipts'] = {'external': '0' * 64}
                else:
                    del parents['v10']['parent_receipt_sha256']
                save_v11_chain(root, path, parents, receipt, report, rewrite_edges=False)
                with self.assertRaises(LedgerError):
                    load_explicit_oof(InputEvidence(root), records, manifest, path)

    def test_v11_missing_root_receipt_anchor_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records, manifest, path, _, _, report = v11_receipt_fixture(root)
            del report['training_receipt_sha256']
            write_json(root / path, report)
            with self.assertRaisesRegex(LedgerError, 'requires its signed training receipt'):
                load_explicit_oof(InputEvidence(root), records, manifest, path)


class ExclusiveOutputTests(unittest.TestCase):
    def ledger(self):
        v, e, p = analyze_video(record(0, [0, 1, 1, 0]), [(1, 2)])
        return {'summary': {'status': 'complete'}, 'videos': [v], 'events': e,
                'predictions': p, 'alias_conflicts': [], 'provenance': {'input_hashes': {'test': 'a' * 64}}}

    def test_json_csv_output_hashes_completion_and_second_write_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'new-ledger'
            write_ledger(destination, self.ledger())
            files = {p.name: p.read_bytes() for p in destination.iterdir()}
            self.assertEqual(len(files), 11)
            completion = json.loads(files['completion.json'])
            for name, digest in completion['output_hashes'].items():
                self.assertEqual(hashlib.sha256(files[name]).hexdigest(), digest)
            with (destination / 'events.csv').open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]['semantic_cause'], 'unknown')
            self.assertEqual(rows[0]['name'], '视频_0.mp4')
            with self.assertRaises(FileExistsError):
                write_ledger(destination, self.ledger())
            self.assertEqual(files, {p.name: p.read_bytes() for p in destination.iterdir()})

    def test_existing_empty_directory_or_file_is_not_claimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory, file = Path(tmp) / 'empty', Path(tmp) / 'existing'
            directory.mkdir()
            file.write_bytes(b'unchanged')
            for destination in (directory, file):
                with self.subTest(path=destination), self.assertRaises(FileExistsError):
                    write_ledger(destination, self.ledger())
            self.assertEqual(file.read_bytes(), b'unchanged')
            self.assertEqual(list(directory.iterdir()), [])

    def test_protected_output_directories_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for path in (EXPERIMENTS['v8'] / 'new-ledger', ORIGINAL / 'new-ledger', Path('code/new-ledger')):
                with self.subTest(path=path), self.assertRaises(LedgerError):
                    check_output_path(root, root / path)
            self.assertEqual(check_output_path(root, root / 'output/new-ledger'), root / 'output/new-ledger')

    def test_cli_existing_output_exits_two_without_reading_or_mutating_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / 'existing'
            existing.mkdir()
            script = Path(__file__).with_name('build_optimization_error_ledger.py')
            result = subprocess.run([sys.executable, '-B', str(script), '--root', str(root),
                                     '--output', str(existing)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('already exists', result.stderr)
            self.assertEqual(list(existing.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
