#!/usr/bin/env python3
"""Fixed two-content+rename fresh product parity, not held-out accuracy."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
from optimized_grouped_training import ROOT,load_grouped,digest
from optimized_locator import load_bundle
from optimized_detector import analyze_optimized_video,predict_record
from verify_recall_diagnostic import check


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['v4','v5','v6','v6r','v7','v8','v9'],required=True);args=p.parse_args()
    root=ROOT/'output'/('algorithm-opt-2026-10-02-'+args.experiment)
    bundlepath=root/'deployment_diagnostic/locator_bundle.json'
    if not bundlepath.is_file():bundlepath=root/'deployment_candidate/locator_bundle.json'
    bundle=load_bundle(bundlepath);out=root/'product_validation';out.mkdir(exist_ok=False)
    _,records,_=load_grouped();byname={r['name']:r for r in records}
    before={n:digest(ROOT/'models'/n) for n in ['optimized_locator_v1.json','optimized_motion_locator_v1.json']}
    result={'role':'development_seen_fresh_pixel_parity_and_speed_only_not_generalization','experiment':args.experiment,
            'diagnostic_only':bundle['diagnostic_only'],'default_promoted':False,'checks':{},'bundle_sha256':digest(bundlepath)}
    started=time.perf_counter()
    try:
        for name in ['0207.mp4','0317.mp4']:
            r=byname[name];folder=out/Path(name).stem;folder.mkdir()
            report=analyze_optimized_video(r['input_path'],folder,bundle_path=bundlepath)
            if bundle['diagnostic_only']:result['checks'][name]=check(report,folder,r,bundle)
            else:
                expect=predict_record(bundle,r);frame=float(np.max(np.abs(expect['frame_probabilities']-report['raw_frame_probabilities'])));video=abs(expect['video_probability']-report['video_evidence_score'])
                if max(frame,video)>=2e-6:raise ValueError('fresh candidate differs')
                if [(s['start_frame'],s['end_frame']) for s in report['segments']]!=expect['intervals']:raise ValueError('fresh intervals differ')
                result['checks'][name]={'frame_probability_max_error':frame,'video_probability_error':video,'elapsed_seconds':report['elapsed_seconds'],'intervals':expect['intervals']}
        r=byname['0207.mp4'];renamed=out/'no_identity_hint.mp4';shutil.copyfile(r['input_path'],renamed)
        folder=out/'renamed';folder.mkdir();report=analyze_optimized_video(renamed,folder,bundle_path=bundlepath)
        if bundle['diagnostic_only']:result['checks']['renamed']=check(report,folder,r,bundle)
        else:
            expect=predict_record(bundle,r);result['checks']['renamed']={'intervals':expect['intervals'],'elapsed_seconds':report['elapsed_seconds']}
        if result['checks']['renamed']['intervals']!=result['checks']['0207.mp4']['intervals']:raise ValueError('filename changed output')
        result['status']='passed'
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        after={n:digest(ROOT/'models'/n) for n in before};result.update(default_model_hashes=after,default_models_unchanged=before==after,elapsed_seconds=time.perf_counter()-started,
            verifier_sha256=digest(__file__),runtime_sha256=digest(ROOT/'code/optimized_detector.py'))
        with (out/'report.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    if before!=after:raise ValueError('default model changed unexpectedly')
    print(json.dumps({'status':result['status'],'experiment':args.experiment,'checks':result['checks']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
