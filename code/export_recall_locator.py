#!/usr/bin/env python3
"""Export an inner-OOF-chosen v2 bundle; never select using outer probabilities."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
from optimized_grouped_training import ROOT,OLD,NEW,RECIPES,load_grouped,fit_member,digest
from optimized_locator import SCHEMA,export_model
from optimized_detector import predict_record
from run_recall_selection import ALL_CANDIDATES,Probabilities,load_fold,blend_probabilities,calibration_modes,calibrate,choose,evaluate,fit_calibration
from run_algorithm_optimization import _group_metrics
from optimized_feature_view import feature_view
from optimized_feature_view_v2 import feature_view_v2


def aggregated_inner(records):
    result={}
    for recipe in RECIPES:
        fp={i:[] for i in range(len(records))};vp={i:[] for i in fp};bp={i:[] for i in fp} if 'boundary' in recipe else None
        for fold in range(5):
            inner,_,_=load_fold(fold,recipe,records)
            for i in inner.frame:
                fp[i].append(inner.frame[i]);vp[i].append(inner.video[i])
                if bp is not None:bp[i].append(inner.boundary[i])
        assert all(len(v)==4 for v in fp.values())
        result[recipe]=Probabilities({i:np.mean(v,axis=0).astype(np.float32) for i,v in fp.items()},
             {i:float(np.mean(v)) for i,v in vp.items()},
             {i:tuple(np.mean(np.stack(v),axis=0)) for i,v in bp.items()} if bp is not None else None)
    return result


def export_member(recipe,records):
    indices=list(range(len(records)));fp,vp,state=fit_member(recipe,records,indices,indices,20261002+9999)
    frame_model,video_model,transform,frames,video,extra=state
    if recipe.endswith('_tcn'):
        fm=deepcopy(frame_model);fm['schema_version']='optimized-temporal-head-v1'
    else:fm=export_model(frame_model)
    member={'recipe':recipe,'weight':1.,'transform':transform,'frame_model':fm,'video_model':export_model(video_model),'boundary_models':extra['boundary_models']}
    member=json.loads(json.dumps(member,allow_nan=False))
    probe={'schema_version':SCHEMA,'members':[member],'recipe':recipe,'decoder':{'threshold':.5}}
    errors={'frame_probability':0.,'video_probability':0.,'frame_feature_view':0.,'video_feature_view':0.}
    for i,r in enumerate(records):
        view=feature_view_v2 if transform.get('feature_view')=='shared-label-free-v2' else feature_view
        x,v,n=view(recipe,r,transform.get('pca'));assert n==transform['frame_feature_names']
        out=predict_record(probe,r)
        for key,a,b in [('frame_probability',out['frame_probabilities'],fp[i]),('video_probability',out['video_probability'],vp[i]),('frame_feature_view',x,frames[i]),('video_feature_view',v,video[i])]:
            e=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));assert e<2e-6,(key,e)
            errors[key]=max(errors[key],e)
    return member,Probabilities(fp,vp,extra['boundary_probabilities']),errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-failed-experiment',action='store_true',help='explicit diagnostic export, never promotes a default')
    args=parser.parse_args()
    selection_path=NEW/'selection_audited/report.json'
    selection=json.loads(selection_path.read_text('utf-8'))
    assert selection['status']=='complete'
    if not selection['statistical_promotion_passed'] and not args.allow_failed_experiment:
        raise ValueError('candidate failed acceptance; explicit --allow-failed-experiment required for non-default diagnostic export')
    # Export may be useful even if the experiment is rejected, but no default
    # promotion is performed here. The calling workflow decides acceptance.
    output=NEW/'deployment_diagnostic';assert not output.exists();output.mkdir()
    started=time.perf_counter();manifest,records,profiles=load_grouped();indices=list(range(len(records)))
    from optimized_training_provenance import collect_new_fit_inputs
    input_evidence=collect_new_fit_inputs()
    inner=aggregated_inner(records)
    # SHA held-out partitions used only for leave-group meta calibration.
    partitions=[{'validation':f} for f in manifest['outer_folds']]
    decisions=[]
    for candidate in ALL_CANDIDATES:
        probabilities=blend_probabilities(candidate,inner,indices);choices=[]
        for mode,p,state in calibration_modes(records,indices,probabilities,partitions):
            row=choose(records,indices,p,True,20261002+899,mode=='joint')
            choices.append({'candidate':candidate.name,'calibration_mode':mode,'calibration':state,**row})
        decision=max(choices,key=lambda r:(r['stable_utility'],*r['key']));decisions.append(decision)
        print(json.dumps({'inner_aggregate':candidate.name,'score':decision['stable_utility']}),flush=True)
    chosen=max(decisions,key=lambda r:(r['stable_utility'],*r['key']))
    fast=max([r for r in decisions if r['candidate'] in ('motion_rf','motion_et','local_motion_et')],key=lambda r:(r['stable_utility'],*r['key']))
    candidates={c.name:c for c in ALL_CANDIDATES};fitted={};probabilities={};parity={}
    selected_recipes=sorted({r for row in [chosen,fast] for r,w in candidates[row['candidate']].members})
    for recipe in selected_recipes:
        fitted[recipe],probabilities[recipe],parity[recipe]=export_member(recipe,records)
    def bundle(row):
        candidate=candidates[row['candidate']];members=[dict(deepcopy(fitted[r]),weight=w) for r,w in candidate.members]
        b={'schema_version':SCHEMA,'recipe':members[0]['recipe'] if len(members)==1 else 'ensemble','members':members,'decoder':row['config'],'calibration':row['calibration'],
           'raw_feature_names':{k:records[0][k+'_names'] for k in ['motion','corrected','local']},'feature_profiles':profiles,
           'training_video_sha256':[r['sha256'] for r in records],'training_video_names':[r['name'] for r in records],
           'training_role':'all inspected 77 video rows / 76 unique byte-identical contents; conflict aliases retained, not blind',
           'deployment_selection':{'role':'inner_OOF_aggregate_only_optimistic_not_validation','uses_outer_OOF_probabilities':False,'candidate':row['candidate']},
           'model_note':'Offline human-review locator. Monotone meta calibration is inner tuning, not independent risk probability.',
           'evaluation_note':'Primary comparison is content-grouped nested selection, not final-fit examples or inner-aggregate ranking.',
           'protocol_sha256':digest(NEW/'protocol_amended.json'),'dataset_manifest_sha256':digest(NEW/'dataset_manifest.json'),
           'comparison_report_sha256':digest(selection_path),
           'diagnostic_only':not selection['statistical_promotion_passed'],
           'default_deployment_acceptance':'accepted_statistically_pending_product' if selection['statistical_promotion_passed'] else 'rejected',
           'statistical_promotion_checks':selection['statistical_promotion_checks']}
        original=blend_probabilities(candidate,probabilities,indices);reference=calibrate(original,row['calibration']);intervals=evaluate(records,indices,reference,row['config']);errors={}
        for i,r in enumerate(records):
            p=predict_record(b,r)
            e=float(np.max(np.abs(p['frame_probabilities']-reference.frame[i])));assert e<2e-6
            assert abs(p['video_probability']-reference.video[i])<2e-6
            assert p['intervals']==intervals[i]
            errors[str(i)]=e
        b['parity']={'max_frame_probability_error':max(errors.values()),'decoded_intervals_equal_for_all_77':True}
        return b
    artifacts={}
    for name,row in [('locator_bundle.json',chosen),('fast_locator_bundle.json',fast)]:
        b=bundle(row);p=output/name
        with p.open('x',encoding='utf-8') as f:json.dump(b,f,ensure_ascii=False,allow_nan=False)
        artifacts[str(p)]=digest(p)
    source_dir=output/'sources';source_dir.mkdir();sources={}
    for name in ['export_recall_locator.py','optimized_detector.py','optimized_feature_view_v2.py','optimized_joint_calibration.py','optimized_probability_calibration.py','optimized_recall_decoder.py','optimized_local_motion.py','optimized_grouped_training.py','run_recall_selection.py','optimized_calibration_state.py','optimized_training_provenance.py']:
        source=ROOT/'code'/name;target=source_dir/name;target.write_bytes(source.read_bytes());sources[str(source)]=digest(source)
    final_input_evidence=collect_new_fit_inputs()
    assert final_input_evidence.inputs_sha256==input_evidence.inputs_sha256
    assert final_input_evidence.sources==input_evidence.sources
    report={'status':'complete','role':'inner_aggregate_diagnostic_export_not_validation',
            'diagnostic_only':not selection['statistical_promotion_passed'],'default_promoted':False,
            'comparison_report_sha256':digest(selection_path),
            'fullfit_inputs_sha256':input_evidence.inputs_sha256,'fullfit_input_hashes':dict(input_evidence.input_hashes),
            'fullfit_source_hashes':dict(input_evidence.sources),
            'fullfit_before_after_inputs_unchanged':True,'chosen':chosen,'fast_chosen':fast,'all_decisions':decisions,
            'fit_parity':parity,'artifact_sha256':artifacts,'source_sha256':sources,'elapsed_seconds':time.perf_counter()-started}
    with (output/'report.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else None)
    print(json.dumps({'phase':'export_complete','chosen':chosen['candidate'],'fast':fast['candidate'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
