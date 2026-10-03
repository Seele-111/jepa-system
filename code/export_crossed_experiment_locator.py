#!/usr/bin/env python3
"""Inner-only export for v8/v9, diagnostic unless all fixed guards passed."""
from __future__ import annotations
import argparse, importlib, json, time
from copy import deepcopy
import numpy as np
import run_compact_experiment_v4 as V4
import run_event_compact_experiment_v8 as V8
from optimized_grouped_training import ROOT, digest, load_grouped
from optimized_event_compact_v8 import fit_event_compact_member, export_event_compact_member
from optimized_normal_cost_v9 import fit_normal_cost_member, export_normal_cost_member
from optimized_compact_model_v4 import fit_compact_member, export_compact_member
from optimized_locator import SCHEMA
from optimized_detector import predict_record
from run_optimization_selection import Probabilities


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=['v8','v9'],required=True);p.add_argument('--allow-failed-experiment',action='store_true');args=p.parse_args()
    version=args.experiment;exp=importlib.import_module('run_event_compact_experiment_v8' if version=='v8' else 'run_normal_cost_experiment_v9')
    start=time.perf_counter();receipt=exp.check_receipt();_,records,profiles=load_grouped();ids=list(range(len(records)))
    report=json.loads((exp.OUT/'report.json').read_text('utf-8'));accepted=bool(report['statistical_promotion_passed'])
    if not accepted and not args.allow_failed_experiment:raise ValueError('failed guards; explicit diagnostic-only export required')
    out=exp.OUT/('deployment_candidate' if accepted else 'deployment_diagnostic');out.mkdir(exist_ok=False)
    recipes=sorted({r for n,m in exp.CANDIDATES for r,w in m});aggregate={};counts={}
    for recipe in recipes:
        ff={i:[] for i in ids};vv={i:[] for i in ids}
        parent=(V4 if version=='v8' else V8) if recipe==exp.CONTROL else exp
        for fold in range(5):
            inner,_,_=parent.load_new_fold(fold,recipe,records,parent.check_receipt())
            for i in inner.frame:ff[i].append(inner.frame[i]);vv[i].append(inner.video[i])
        if any(len(v)!=4 for v in ff.values()):raise ValueError('inner aggregate coverage mismatch')
        counts[recipe]=[len(ff[i]) for i in ids]
        aggregate[recipe]=Probabilities({i:np.mean(v,axis=0).astype(np.float32) for i,v in ff.items()}, {i:float(np.mean(v)) for i,v in vv.items()},None)
    decisions=[]
    for name,members in exp.CANDIDATES:
        row=exp.choose(records,ids,exp.blended(members,aggregate,ids),20261002+777);row['candidate']=name;decisions.append(row)
    chosen=max(decisions,key=lambda z:(z['stable_utility'],*z['key'][1:],-[n for n,m in exp.CANDIDATES].index(z['candidate'])))
    wanted=dict(exp.CANDIDATES)[chosen['candidate']];members=[];reference={};errors={}
    before={n:digest(ROOT/'models'/n) for n in ['optimized_locator_v1.json','optimized_motion_locator_v1.json']}
    for recipe,w in wanted:
        if recipe.startswith('normalcost_'):fit,export=fit_normal_cost_member,export_normal_cost_member
        elif recipe.startswith('event_compact_'):fit,export=fit_event_compact_member,export_event_compact_member
        else:fit,export=fit_compact_member,export_compact_member
        fp,vp,state=fit(recipe,records,ids,ids,20261002+9999,full_fit=True)
        member=json.loads(json.dumps(export(state),allow_nan=False));member['weight']=w;members.append(member)
        reference[recipe]=Probabilities(fp,vp,None);mf=mv=0.
        for i,r in enumerate(records):
            pred=predict_record({'members':[dict(deepcopy(member),weight=1.)],'decoder':{'threshold':.5},'calibration':None},r)
            f=float(np.max(np.abs(pred['frame_probabilities']-fp[i])));v=abs(pred['video_probability']-vp[i]);mf=max(mf,f);mv=max(mv,v)
            if max(f,v)>2e-6:raise ValueError('crossed member sklearn/portable differs')
        errors[recipe]={'frame':mf,'video':mv}
    ref=exp.blended(wanted,reference,ids);expected=exp.decode_predictions(records,ids,ref,chosen['config'])
    bundle={'schema_version':SCHEMA,'recipe':members[0]['recipe'] if len(members)==1 else 'ensemble','members':members,
            'calibration':None,'decoder':chosen['config'],
            'raw_feature_names':{k:records[0][k+'_names'] for k in ['motion','corrected','local']},'feature_profiles':profiles,
            'training_video_sha256':[r['sha256'] for r in records],'training_video_names':[r['name'] for r in records],
            'training_role':'all77 inspected development annotations/76 content,aliases/conflict retained; not blind',
            'model_note':'Offline human review; scores are uncalibrated learned evidence, not physical truth/risk probabilities.',
            'evaluation_note':'Repeated same-development nested primary only; final fullfit and inner aggregates are not validation.',
            'experiment_id':version,'protocol_sha256':digest(exp.OUT/'protocol.json'),'comparison_report_sha256':digest(exp.OUT/'report.json'),
            'diagnostic_only':not accepted,'default_deployment_acceptance':'accepted_statistically_pending_product' if accepted else 'rejected',
            'statistical_promotion_checks':report['statistical_promotion_checks'],
            'deployment_selection':{'role':'inner_aggregate_only_optimistic_not_validation','candidate':chosen['candidate'],'uses_outer_OOF_probabilities':False}}
    mf=mv=0.
    for i,r in enumerate(records):
        pred=predict_record(bundle,r);f=float(np.max(np.abs(pred['frame_probabilities']-ref.frame[i])));v=abs(pred['video_probability']-ref.video[i])
        if max(f,v)>2e-6 or pred['intervals']!=expected[i]:raise ValueError('full crossed bundle parity differs')
        mf=max(mf,f);mv=max(mv,v)
    bundle['parity']={'rows':len(records),'max_frame_probability_error':mf,'max_video_probability_error':mv,'all_decoded_intervals_equal':True}
    exp.check_receipt();after={n:digest(ROOT/'models'/n) for n in before}
    if before!=after:raise ValueError('default model changed during diagnostic export')
    exp.write_new(out/'locator_bundle.json',bundle)
    sources=['export_crossed_experiment_locator.py','optimized_detector.py','optimized_event_compact_v8.py','optimized_normal_cost_v9.py','optimized_compact_model_v4.py','optimized_compact_features_v4.py','optimized_event_training_v5.py','optimized_duration_decoder_v4.py']
    for n in sources:
        target=out/'sources'/n;target.parent.mkdir(exist_ok=True)
        with target.open('xb') as f:f.write((ROOT/'code'/n).read_bytes())
    result={'status':'complete','role':'inner_only_deployment_export_not_validation','diagnostic_only':not accepted,'default_promoted':False,
            'chosen':chosen,'all_decisions':decisions,'inner_coverage':counts,'fullfit_training_receipt_sha256':receipt['receipt_sha256'],
            'before_after_inputs_and_sources_equal':True,'member_portable_parity':errors,'bundle_portable_parity':bundle['parity'],
            'bundle_sha256':digest(out/'locator_bundle.json'),'source_sha256':{n:digest(ROOT/'code'/n) for n in sources},
            'default_models_unchanged':True,'elapsed_seconds':time.perf_counter()-start}
    exp.write_new(out/'report.json',result);print(json.dumps({'status':'complete','experiment':version,'diagnostic_only':not accepted,'candidate':chosen['candidate'],'config':chosen['config'],'parity':bundle['parity']}),flush=True)

if __name__=='__main__':main()
