#!/usr/bin/env python3
"""Fresh diagnostic candidate smoke; never a held-out metric or default promotion."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from optimized_grouped_training import ROOT,NEW,load_grouped,digest
from optimized_locator import load_bundle
from optimized_detector import analyze_optimized_video,predict_record,required_channels,from_wsl


def check(report,root,reference,bundle):
    expected=predict_record(bundle,reference)
    pe=float(np.max(np.abs(expected['frame_probabilities']-report['raw_frame_probabilities'])))
    ve=abs(expected['video_probability']-report['video_evidence_score'])
    assert pe<2e-6 and ve<2e-6
    intervals=[(s['start_frame'],s['end_frame']) for s in report['segments']]
    assert intervals==expected['intervals']
    assert report['diagnostic_only'] is True and report['default_deployment_acceptance']=='rejected'
    assert any('未通过默认部署验收' in warning for warning in report['warnings'])
    assert report['seen_in_development'] is True and report['video_encoding']['browser_playable']
    assert report['video_encoding']['codec']=='h264'
    channels=required_channels(bundle);worker=report['feature_provenance'].get('feature_worker')
    if 'corrected' in channels:
        assert report['jepa_usage']['vjepa']==report['jepa_usage']['ijepa']=='true_masked_predictor'
        assert worker['feature_cache_used'] is False and worker['status']=='ok'
        assert set(worker['features'])==channels-{'motion','local'}
        for channel,entry in worker['features'].items():
            p=from_wsl(entry['npz_path']).resolve();assert p.is_relative_to(root.resolve())
            with np.load(p,allow_pickle=False) as data:
                np.testing.assert_array_equal(data[entry['feature_key']],reference[channel])
    else:
        assert worker is None
        assert report['jepa_usage']['vjepa']==report['jepa_usage']['ijepa']=='not_used'
    if 'local' in channels:
        assert report['feature_provenance']['local']['extraction_fresh'] is True
        with np.load(root/'local_motion_features.npz',allow_pickle=False) as data:
            x=np.concatenate([data['signals'],data['feature_valid'].astype(np.float32)],axis=1)
            np.testing.assert_array_equal(x,reference['local'])
    cap=cv2.VideoCapture(str(root/'annotated.mp4'));n=0
    while True:
        ok,_=cap.read()
        if not ok:break
        n+=1
    cap.release();assert n==len(reference['labels'])
    return {'frame_probability_max_error':pe,'video_probability_error':ve,'intervals':intervals,
            'elapsed_seconds':report['elapsed_seconds'],'total_frames':n,'codec':'h264',
            'channels':sorted(channels),'feature_cache_used':False if worker else None,
            'local_fresh':bool('local' in channels),'seen_in_development':True,
            'role':'functional_parity_and_speed_only_not_generalization'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=NEW/'product_diagnostic/validation.json');args=parser.parse_args()
    if args.output.exists():raise FileExistsError('refusing to overwrite diagnostic validation')
    args.output.parent.mkdir(parents=True,exist_ok=True);runs=args.output.parent/(args.output.stem+'_fresh')
    runs.mkdir(exist_ok=False);_,records,_=load_grouped();by_name={r['name']:r for r in records}
    defaults=[ROOT/'models/optimized_locator_v1.json',ROOT/'models/optimized_motion_locator_v1.json'];before={str(p):digest(p) for p in defaults}
    bundles={name:load_bundle(NEW/'deployment_diagnostic'/filename) for name,filename in [('primary','locator_bundle.json'),('fast','fast_locator_bundle.json')]}
    assert all(b['diagnostic_only'] and b['default_deployment_acceptance']=='rejected' for b in bundles.values())
    result={'schema_version':'diagnostic-recall-product-validation-v2','created_at':datetime.now().astimezone().isoformat(),
            'role':'development_seen_diagnostic_smoke_not_blind','diagnostic_only':True,'default_promoted':False,'checks':{}}
    started=time.perf_counter()
    try:
        for mode,bundle in bundles.items():
            for name in ['0207.mp4','0317.mp4']:
                record=by_name[name];root=runs/(mode+'_'+Path(name).stem);root.mkdir()
                report=analyze_optimized_video(record['input_path'],root,bundle_path=NEW/'deployment_diagnostic'/('locator_bundle.json' if mode=='primary' else 'fast_locator_bundle.json'),algorithm='optimized' if mode=='primary' else 'optimized_fast')
                result['checks'][mode+'_'+name]=check(report,root,record,bundle)
        record=by_name['0207.mp4'];renamed=runs/'name_independent_probe.mp4';shutil.copyfile(record['input_path'],renamed)
        root=runs/'fast_renamed';root.mkdir();bundle=bundles['fast']
        report=analyze_optimized_video(renamed,root,bundle_path=NEW/'deployment_diagnostic/fast_locator_bundle.json',algorithm='optimized_fast')
        result['checks']['fast_renamed']=check(report,root,record,bundle)
        assert result['checks']['fast_renamed']['intervals']==result['checks']['fast_0207.mp4']['intervals']
        result['status']='passed'
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        after={str(p):digest(p) for p in defaults};result['default_model_hashes']=after;result['default_models_unchanged']=before==after
        result['elapsed_seconds']=time.perf_counter()-started;result['verifier_sha256']=digest(__file__)
        with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    assert before==after
    print(json.dumps({'status':result['status'],'default_promoted':False,'checks':result['checks']},ensure_ascii=False))

if __name__=='__main__':main()
