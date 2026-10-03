#!/usr/bin/env python3
"""Focused optimizer/runtime tests plus 154 default output compatibility checks."""
import argparse,importlib.util,json,time,unittest
from pathlib import Path
import numpy as np
from optimized_grouped_training import ROOT,load_grouped,digest
from optimized_locator import load_bundle
import optimized_detector

MODULES=['test_feature_blocks_v10','test_proposal_review_v11','test_compact_boundary_v12',
    'test_spatial_jepa_v13','test_spatial_jepa_v13r','test_optimized_error_ledger','test_audit_feature_blocks_v10',
    'test_audit_spatial_jepa_v13','test_experimental_runtime_v10_v11',
    'test_optimized_detector','test_optimized_compact_v4','test_optimized_duration_decoder_v4',
    'test_optimized_boundary_head','test_optimized_event_compact_v8']
DEFAULTS={'optimized_locator_v1.json':'093c14ec9b2ffc91dea776f694a8278bbd2f4d66a1881fe10d8399fc24301dbd',
    'optimized_motion_locator_v1.json':'77e5d7eeb63653c6ddc0ea33da8e455c5b737d46f87c1cfde96c43a1381d5670'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refuse validation overwrite')
    started=time.perf_counter();sources={n:digest(ROOT/'code'/(n+'.py')) for n in MODULES+['optimized_detector']}
    suite=unittest.TestLoader().loadTestsFromNames(MODULES);result=unittest.TextTestRunner(verbosity=1).run(suite)
    row={'status':'passed' if result.wasSuccessful() else 'failed','role':'focused_tests_and_default_regression_not_accuracy',
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':[str(t) for t,r in result.skipped],
        'modules':MODULES,'source_sha256':sources,'default_compatibility':None}
    if result.wasSuccessful():
        before=ROOT/'output/algorithm-opt-v10-blocks/runtime_before/optimized_detector.py'
        spec=importlib.util.spec_from_file_location('optimizer_original_detector',before);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
        _,records,_=load_grouped();count=0
        for name,h in DEFAULTS.items():
            path=ROOT/'models'/name
            if digest(path)!=h:raise ValueError('default model changed')
            bundle=load_bundle(path)
            for r in records:
                old=original.predict_record(bundle,r);new=optimized_detector.predict_record(bundle,r)
                if set(old)!=set(new) or not all(np.array_equal(old[k],new[k]) for k in ['frame_probabilities','score']) or old['video_probability']!=new['video_probability'] or old['intervals']!=new['intervals']:
                    raise ValueError('default output changed')
                count+=1
        row['default_compatibility']={'rows':count,'probabilities_scores_bitwise_equal':True,
            'video_probabilities_intervals_keys_equal':True,'model_sha256':DEFAULTS,
            'before_runtime_sha256':digest(before),'after_runtime_sha256':digest(ROOT/'code/optimized_detector.py')}
    row['before_after_sources_unchanged']=all(digest(ROOT/'code'/(n+'.py'))==h for n,h in sources.items())
    if not row['before_after_sources_unchanged']:row['status']='failed'
    row['elapsed_seconds']=time.perf_counter()-started
    with a.output.open('x',encoding='utf-8') as f:json.dump(row,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':row['status'],'tests_run':row['tests_run'],'skipped':len(row['skipped']),'default_checks':(row['default_compatibility'] or {}).get('rows',0)}))
    if row['status']!='passed':raise SystemExit(1)
if __name__=='__main__':main()
