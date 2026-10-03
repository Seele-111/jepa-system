#!/usr/bin/env python3
"""Fresh-pixel product parity over representative seen content, not a blind test."""
import argparse,json,shutil,time
from pathlib import Path
from urllib.request import build_opener,ProxyHandler
from optimized_grouped_training import ROOT,load_grouped,digest
from optimized_locator import load_bundle
from optimized_detector import analyze_optimized_video
from verify_recall_diagnostic import check


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['v10','v11'],required=True);a=p.parse_args()
    root=ROOT/'output'/('algorithm-opt-v10-blocks' if a.experiment=='v10' else 'algorithm-opt-v11-proposal-review')
    bundle_path=root/'deployment_diagnostic/locator_bundle.json';bundle=load_bundle(bundle_path)
    if not bundle['diagnostic_only']:raise ValueError('this verifier is diagnostic-only')
    out=root/'product_validation';out.mkdir(exist_ok=False);_,records,_=load_grouped();byname={r['name']:r for r in records}
    defaults={n:digest(ROOT/'models'/n) for n in ('optimized_locator_v1.json','optimized_motion_locator_v1.json')}
    result={'experiment':a.experiment,'role':'development_seen_fresh_pixel_parity_only_not_generalization',
        'diagnostic_only':True,'default_promoted':False,'bundle_sha256':digest(bundle_path),'checks':{}}
    start=time.perf_counter()
    try:
        health={}
        opener=build_opener(ProxyHandler({}))
        for port in (5002,5004):
            with opener.open('http://127.0.0.1:'+str(port)+'/health',timeout=5) as response:health[str(port)]=json.load(response)
        if health['5004']['busy']:raise RuntimeError('worker busy; do not interfere')
        result['initial_health']=health
        for name in ('0207.mp4','0317.mp4','0353.mp4'):
            r=byname[name];folder=out/Path(name).stem;folder.mkdir()
            report=analyze_optimized_video(r['input_path'],folder,bundle_path=bundle_path)
            result['checks'][name]=check(report,folder,r,bundle)
        r=byname['0207.mp4'];renamed=out/'no_identity_hint.mp4';shutil.copyfile(r['input_path'],renamed)
        folder=out/'renamed';folder.mkdir();report=analyze_optimized_video(renamed,folder,bundle_path=bundle_path)
        result['checks']['renamed']=check(report,folder,r,bundle)
        if result['checks']['renamed']['intervals']!=result['checks']['0207.mp4']['intervals']:raise ValueError('rename changed prediction')
        result['status']='passed'
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        after={n:digest(ROOT/'models'/n) for n in defaults};result.update(default_models=after,default_models_unchanged=defaults==after,
            elapsed_seconds=time.perf_counter()-start,verifier_sha256=digest(__file__),runtime_sha256=digest(ROOT/'code/optimized_detector.py'))
        with (out/'report.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    if defaults!=after:raise ValueError('default model changed')
    print(json.dumps({'status':result['status'],'experiment':a.experiment,'checks':result['checks']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
