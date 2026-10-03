#!/usr/bin/env python3
"""All-content bitwise default regression after opt-in research additions."""
import importlib.util, json, time
import numpy as np
from optimized_grouped_training import ROOT, load_grouped, digest
from optimized_locator import load_bundle
from optimized_detector import predict_record
from run_compact_experiment_v4 import write_new

REFERENCE=ROOT/'output/algorithm-opt-2026-10-02-v2/final_runtime_sources/optimized_detector.py'
PINS={'optimized_locator_v1.json':'093c14ec9b2ffc91dea776f694a8278bbd2f4d66a1881fe10d8399fc24301dbd',
      'optimized_motion_locator_v1.json':'77e5d7eeb63653c6ddc0ea33da8e455c5b737d46f87c1cfde96c43a1381d5670'}


def main():
    start=time.perf_counter();before={n:digest(ROOT/'models'/n) for n in PINS}
    if before!=PINS:raise ValueError('default model fingerprints changed before compatibility check')
    spec=importlib.util.spec_from_file_location('pre_extended_runtime',REFERENCE);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    _,records,_=load_grouped();rows=[]
    for name in PINS:
        bundle=load_bundle(ROOT/'models'/name)
        for i,record in enumerate(records):
            a,b=old.predict_record(bundle,record),predict_record(bundle,record)
            for key in ['frame_probabilities','score']:
                if not np.array_equal(a[key],b[key]):raise ValueError('default frame/effective score changed')
            if a['video_probability']!=b['video_probability'] or a['intervals']!=b['intervals']:raise ValueError('default video/interval changed')
        rows.append({'model':name,'rows':len(records),'bitwise_frame_and_score_equal':True,'video_and_intervals_equal':True})
    after={n:digest(ROOT/'models'/n) for n in PINS}
    if before!=after:raise ValueError('default model changed during check')
    result={'status':'passed','checks':rows,'default_model_sha256':after,'reference_source_sha256':digest(REFERENCE),
            'current_runtime_sha256':digest(ROOT/'code/optimized_detector.py'),'verifier_sha256':digest(__file__),
            'elapsed_seconds':time.perf_counter()-start,'role':'compatibility_only_not_new_accuracy_validation'}
    write_new(ROOT/'output/algorithm-opt-2026-10-02-v8/default_compatibility.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
