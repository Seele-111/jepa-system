#!/usr/bin/env python3
"""Only-inner-chosen experiment export, explicitly diagnostic on failed guards."""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import time
import numpy as np
from optimized_grouped_training import ROOT, digest, load_grouped
from optimized_locator import SCHEMA
from optimized_detector import predict_record
from optimized_compact_model_v4 import fit_compact_member, export_compact_member
from optimized_event_training_v5 import fit_event_member, export_event_member
from export_recall_locator import export_member as export_old_member
from run_recall_selection import load_fold
from run_optimization_selection import Probabilities


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['v4','v5'],required=True)
    parser.add_argument('--allow-failed-experiment',action='store_true')
    args=parser.parse_args()
    version=args.experiment
    exp=importlib.import_module('run_compact_experiment_v4' if version=='v4' else 'run_event_experiment_v5')
    chooser=exp.choose_v4 if version=='v4' else exp.choose_v5
    started=time.perf_counter();receipt=exp.check_receipt();_,records,profiles=load_grouped()
    report=json.loads((exp.OUT/'report.json').read_text('utf-8'))
    accepted=bool(report['statistical_promotion_passed'])
    if not accepted and not args.allow_failed_experiment:
        raise ValueError('failed acceptance; require explicit --allow-failed-experiment for diagnostic export')
    output=exp.OUT/('deployment_candidate' if accepted else 'deployment_diagnostic');output.mkdir(exist_ok=False)
    indices=list(range(len(records)));recipes=sorted({r for name,members in exp.CANDIDATES for r,w in members})
    aggregate={};aggregation_counts={}
    for recipe in recipes:
        ff={i:[] for i in indices};vv={i:[] for i in indices}
        for fold in range(5):
            if recipe in exp.RECIPES:
                inner,_,_=exp.load_new_fold(fold,recipe,records,receipt)
            else:inner,_,_=load_fold(fold,recipe,records)
            for i in inner.frame:
                ff[i].append(inner.frame[i]);vv[i].append(inner.video[i])
        if any(len(v)!=4 for v in ff.values()):raise ValueError('inner aggregate coverage requires exactly4 per row')
        aggregation_counts[recipe]=[len(ff[i]) for i in indices]
        aggregate[recipe]=Probabilities({i:np.mean(v,axis=0).astype(np.float32) for i,v in ff.items()},
                                       {i:float(np.mean(v)) for i,v in vv.items()},None)
    decisions=[]
    for name,members in exp.CANDIDATES:
        decision=chooser(records,indices,exp.blended(members,aggregate,indices),20261002+777)
        decision['candidate']=name;decisions.append(decision)
    chosen=max(decisions,key=lambda z:(z['stable_utility'],*z['key'][1:],-[n for n,m in exp.CANDIDATES].index(z['candidate'])))
    wanted=dict(exp.CANDIDATES)[chosen['candidate']];fitted={};reference={};member_errors={}
    for recipe,w in wanted:
        if recipe in exp.RECIPES:
            fit=fit_compact_member if version=='v4' else fit_event_member
            export=export_compact_member if version=='v4' else export_event_member
            fp,vp,state=fit(recipe,records,indices,indices,20261002+9999,full_fit=True)
            member=export(state);member=json.loads(json.dumps(member,allow_nan=False))
            max_f=max_v=0.
            probe={'members':[dict(deepcopy(member),weight=1.)],'calibration':None,
                   'decoder':{'threshold':.5}}
            for i,r in enumerate(records):
                actual=predict_record(probe,r)
                e=float(np.max(np.abs(actual['frame_probabilities']-fp[i])));v=abs(actual['video_probability']-vp[i])
                if max(e,v)>=2e-6:raise ValueError('new fitted/portable member parity differs')
                max_f=max(max_f,e);max_v=max(max_v,v)
            member_errors[recipe]={'frame':max_f,'video':max_v};reference[recipe]=Probabilities(fp,vp,None)
        else:
            member,reference[recipe],member_errors[recipe]=export_old_member(recipe,records)
        fitted[recipe]=member
    members=[dict(deepcopy(fitted[r]),weight=w) for r,w in wanted]
    bundle={'schema_version':SCHEMA,'recipe':members[0]['recipe'] if len(members)==1 else 'ensemble',
            'members':members,'calibration':None,'decoder':chosen['config'],
            'raw_feature_names':{k:records[0][k+'_names'] for k in ['motion','corrected','local']},
            'feature_profiles':profiles,'training_video_sha256':[r['sha256'] for r in records],
            'training_video_names':[r['name'] for r in records],
            'training_role':'all inspected77 annotation rows/76 unique contents; duplicate conflict retained; not blind',
            'model_note':'Offline human-review candidate; evidence scores are uncalibrated, not risk probabilities.',
            'evaluation_note':'Only content-grouped nested primary decides acceptance. Final fit/inner average are not accuracy evidence.',
            'experiment_id':version,'protocol_sha256':digest(exp.OUT/'protocol.json'),
            'comparison_report_sha256':digest(exp.OUT/'report.json'),
            'diagnostic_only':not accepted,'default_deployment_acceptance':'accepted_statistically_pending_product' if accepted else 'rejected',
            'statistical_promotion_checks':report['statistical_promotion_checks'],
            'deployment_selection':{'role':'inner_aggregate_only_optimistic_not_validation','candidate':chosen['candidate'],
                                    'uses_outer_OOF_probabilities':False}}
    reference_probs=exp.blended(wanted,reference,indices)
    expected=exp.decode_predictions(records,indices,reference_probs,chosen['config'])
    max_frame=max_video=0.
    for i,r in enumerate(records):
        p=predict_record(bundle,r)
        f=float(np.max(np.abs(p['frame_probabilities']-reference_probs.frame[i])));v=abs(p['video_probability']-reference_probs.video[i])
        if max(f,v)>=2e-6 or p['intervals']!=expected[i]:raise ValueError('full bundle portable parity differs')
        max_frame=max(max_frame,f);max_video=max(max_video,v)
    bundle['parity']={'rows':len(records),'max_frame_probability_error':max_frame,'max_video_probability_error':max_video,'all_decoded_intervals_equal':True}
    exp.check_receipt()
    exp.write_new(output/'locator_bundle.json',bundle)
    source_names=['export_new_experiment_locator.py','optimized_detector.py','optimized_compact_features_v4.py','optimized_compact_model_v4.py','optimized_event_training_v5.py','optimized_duration_decoder_v4.py']
    sourcehash={n:digest(ROOT/'code'/n) for n in source_names}
    for n in source_names:
        target=output/'sources'/n;target.parent.mkdir(exist_ok=True)
        with target.open('xb') as f:f.write((ROOT/'code'/n).read_bytes())
    export_report={'status':'complete','role':'inner_aggregate_diagnostic_export_not_validation','diagnostic_only':not accepted,
                   'default_promoted':False,'chosen':chosen,'all_decisions':decisions,'inner_coverage':aggregation_counts,
                   'fullfit_training_receipt_sha256':receipt['receipt_sha256'],'before_after_inputs_and_sources_equal':True,
                   'member_portable_parity':member_errors,'bundle_portable_parity':bundle['parity'],
                   'bundle_sha256':digest(output/'locator_bundle.json'),'source_sha256':sourcehash,
                   'elapsed_seconds':time.perf_counter()-started}
    exp.write_new(output/'report.json',export_report)
    print(json.dumps({'status':'complete','diagnostic_only':not accepted,'candidate':chosen['candidate'],'config':chosen['config'],'parity':bundle['parity'],'seconds':export_report['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
