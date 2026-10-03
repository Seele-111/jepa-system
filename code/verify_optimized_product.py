#!/usr/bin/env python3
"""Fresh local product smoke and training/live parity; not a generalization score."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, build_opener, ProxyHandler
import uuid

import cv2
import numpy as np
from demo_detector import analyze_video
from optimized_detector import from_wsl, predict_record, source_digest
from optimized_locator import load_bundle
from run_optimization_selection import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'output' / 'algorithm-opt-2026-10-02'
BASE = 'http://127.0.0.1:5002'
OPENER = build_opener(ProxyHandler({}))


def request(path, *, data=None, headers=None):
    return OPENER.open(Request(BASE + path, data=data, headers=headers or {}), timeout=900)


def json_request(path, **kwargs):
    with request(path, **kwargs) as response:
        return response.status, json.load(response)


def completed(job_id):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        _, status = json_request('/api/status/' + job_id)
        if status['status'] == 'error':
            raise AssertionError('product job failed: ' + str(status.get('error')))
        if status['status'] == 'done':
            return json_request('/api/result/' + job_id)[1]
        time.sleep(.5)
    raise TimeoutError('product smoke job timed out')


def transport_checks(base_path):
    checks = {}
    for kind in ('annotated', 'original'):
        with request(base_path + '/' + kind, headers={'Range': 'bytes=0-1023'}) as response:
            content = response.read()
            assert response.status == 206 and len(content) == 1024
            assert response.headers['Content-Range'].startswith('bytes 0-1023/')
            checks[kind + '_range'] = {'status': response.status, 'bytes': len(content),
                                       'content_range': response.headers['Content-Range']}
    with request(base_path + '/report') as response:
        assert response.status == 200 and 'attachment' in response.headers['Content-Disposition']
        payload = json.load(response)
        assert payload['schema_version'] == 'jepa-demo-report-v1'
        checks['report_download'] = {'status': 200, 'attachment': True}
    try:
        request(base_path + '/motion_features.npz')
    except Exception as exc:
        assert getattr(exc, 'code', None) == 404
        checks['private_features_blocked'] = True
    else:
        raise AssertionError('private features unexpectedly exposed')
    return checks


def verify_report(report, root, reference, bundle):
    assert report['schema_version'] == 'jepa-demo-report-v1'
    assert report['model_bundle_sha256'] == source_digest(report['model_bundle_path'])
    assert report['seen_in_development'] is True
    assert report['video_encoding']['codec'] == 'h264'
    assert report['video_encoding']['browser_playable'] is True
    assert report['jepa_usage']['proxy_fallback'] is False
    expected = predict_record(bundle, reference)
    fp_error = float(np.max(np.abs(expected['frame_probabilities'] - report['raw_frame_probabilities'])))
    vp_error = abs(expected['video_probability'] - report['video_evidence_score'])
    assert fp_error < 2e-6 and vp_error < 2e-6
    assert expected['intervals'] == [(s['start_frame'], s['end_frame']) for s in report['segments']]
    worker = report['feature_provenance'].get('feature_worker')
    if report['algorithm'] == 'optimized':
        assert report['method'] == 'optimized_true_jepa_motion'
        assert report['jepa_usage']['vjepa'] == 'true_masked_predictor'
        assert report['jepa_usage']['ijepa'] == 'true_masked_predictor'
        assert worker['feature_cache_used'] is False and worker['status'] == 'ok'
        assert set(worker['features']) == {'rgb', 'corrected'}
        for channel, info in worker['features'].items():
            path = from_wsl(info['npz_path']).resolve()
            assert path.is_relative_to(root.resolve())
            with np.load(path, allow_pickle=False) as archive:
                values = archive[info['feature_key']]
                assert values.shape == tuple(info['shape']) and len(values) == report['total_frames']
                assert np.isfinite(values).all()
                np.testing.assert_array_equal(values, reference[channel])
    else:
        assert report['algorithm'] == 'optimized_fast'
        assert report['method'] == 'optimized_motion' and worker is None
        assert report['jepa_usage']['vjepa'] == report['jepa_usage']['ijepa'] == 'not_used'
    video = cv2.VideoCapture(str(root / 'annotated.mp4'))
    frames = 0
    while True:
        ok, _ = video.read()
        if not ok:
            break
        frames += 1
    video.release()
    assert frames == report['total_frames']
    return {'report_path': str(root / 'demo_report.json'), 'algorithm': report['algorithm'], 'method': report['method'],
            'elapsed_seconds': report['elapsed_seconds'], 'segment_count': report['segment_count'],
            'intervals': expected['intervals'], 'frame_probability_parity_error': fp_error,
            'video_probability_parity_error': vp_error, 'annotated_decoded_frames': frames,
            'codec': 'h264', 'seen_in_development': True,
            'feature_cache_used': False if worker else None,
            'feature_transport': worker['transport'] if worker else 'CPU_only_not_JEPA'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', action='append', default=[])
    parser.add_argument('--output', type=Path, default=DATA / 'product_smoke' / 'validation.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('refusing to overwrite product validation')
    started = time.monotonic()
    manifest, records, _, _ = load_dataset(DATA)
    by_name = {row['name']: records[i] for i, row in enumerate(manifest['rows'])}
    primary = load_bundle(ROOT / 'models' / 'optimized_locator_v1.json')
    fast = load_bundle(ROOT / 'models' / 'optimized_motion_locator_v1.json')
    result = {'schema_version': 'optimized-product-validation-v1', 'created_at': datetime.now().astimezone().isoformat(),
              'role': 'development_seen_product_smoke_not_blind', 'checks': {}}
    try:
        status, health = json_request('/health')
        assert status == 200 and health['service'] == 'jepa-demo'
        for name, folder in [('0207.mp4', 'final_0207'), ('0317.mp4', 'final_0317')]:
            root = DATA / 'product_smoke' / folder
            report = json.loads((root / 'demo_report.json').read_text('utf-8'))
            result['checks'][folder] = verify_report(report, root, by_name[name], primary)
            result['checks'][folder]['http'] = transport_checks('/samples/' + name.removesuffix('.mp4'))
        fast_root = args.output.parent / (args.output.stem + '_fresh') / '0207_fast'
        assert not fast_root.exists(), 'refusing to reuse fast smoke output'
        source = Path(manifest['rows'][next(i for i, x in enumerate(manifest['rows']) if x['name'] == '0207.mp4')]['input_path'])
        report = analyze_video(source, fast_root, algorithm='optimized_fast')
        result['checks']['final_0207_fast'] = verify_report(report, fast_root, by_name['0207.mp4'], fast)
        for job in args.job:
            report = completed(job)
            root = ROOT / 'output' / 'demo-runs' / job
            result['checks']['ui_' + job] = verify_report(report, root, by_name[report['video_name']], primary)
            result['checks']['ui_' + job]['http'] = transport_checks('/files/' + job)
        # Rename and upload identical content: no filename or annotation is an input feature.
        boundary = uuid.uuid4().hex
        prefix = ('--' + boundary + '\r\nContent-Disposition: form-data; name="algorithm"\r\n\r\noptimized\r\n'
                  '--' + boundary + '\r\nContent-Disposition: form-data; name="video"; filename="renamed_probe.mp4"\r\n'
                  'Content-Type: video/mp4\r\n\r\n').encode()
        payload = prefix + source.read_bytes() + ('\r\n--' + boundary + '--\r\n').encode()
        status, queued = json_request('/api/upload', data=payload, headers={'Content-Type': 'multipart/form-data; boundary=' + boundary})
        assert status == 202
        job = queued['job_id']; report = completed(job)
        assert report['video_name'] == 'renamed_probe.mp4'
        root = ROOT / 'output' / 'demo-runs' / job
        result['checks']['renamed_upload'] = verify_report(report, root, by_name['0207.mp4'], primary)
        result['checks']['renamed_upload'].update(job_id=job, http=transport_checks('/files/' + job))
        status, queued = json_request('/api/samples/0317/analyze', data=urlencode({'algorithm': 'optimized_fast'}).encode())
        assert status == 202
        job = queued['job_id']; report = completed(job)
        result['checks']['fast_api_0317'] = verify_report(report, ROOT / 'output' / 'demo-runs' / job, by_name['0317.mp4'], fast)
        result['checks']['fast_api_0317']['job_id'] = job
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        result['verifier_sha256'] = source_digest(__file__)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'status': result['status'], 'checks': list(result['checks']), 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
