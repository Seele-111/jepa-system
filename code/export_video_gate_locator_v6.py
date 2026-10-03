#!/usr/bin/env python3
"""Inner-only v6 final export; no accepted/failed artifact replaces default here."""
import argparse,json,time
from copy import deepcopy
import numpy as np
import run_video_gate_experiment_v6 as V6
from optimized_grouped_training import ROOT,load_grouped,digest
from optimized_event_training_v5 import fit_event_member,export_event_member
from optimized_video_gate_v6 import GATE_KINDS,fit_gate,apply_gate
from optimized_detector import predict_record
from optimized_locator import SCHEMA
from run_optimization_selection import Probabilities


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--allow-failed-experiment',action='store_true');args=parser.parse_args()
    start=time.perf_counter();receipt=V6.check_receipt();_,records,profiles=load_grouped();ids=list(range(len(records)))
    report=json.loads((V6.OUT/'report.json').read_text('utf-8'));accepted=bool(report['statistical_promotion_passed'])
    if not accepted and not args.allow_failed_experiment:raise ValueError('only explicit diagnostic export authorized for failed pipeline')
    output=V6.OUT/('deployment_candidate' if accepted else 'deployment_diagnostic');output.mkdir(exist_ok=False)
    ff={i:[] for i in ids};vv={c:{i:[] for i in ids} for c in V6.CANDIDATES}
    for fold in range(5):
        inner,_,_,_=V6.load_fold(fold,records,receipt)
        for i in inner['ungated_control'].frame:
            ff[i].append(inner['ungated_control'].frame[i])
            for c in V6.CANDIDATES:vv[c][i].append(inner[c].video[i])
    if any(len(x)!=4 for x in ff.values()):raise ValueError('inner aggregate coverage')
    frame={i:np.mean(x,axis=0).astype(np.float32) for i,x in ff.items()};probs={c:Probabilities(frame,{i:float(np.mean(x)) for i,x in vv[c].items()},None) for c in V6.CANDIDATES}
    decisions=[]
    for c in V6.CANDIDATES:
        row=V6.choose(records,ids,probs[c],20261002+777);row['candidate']=c;decisions.append(row)
    chosen=max(decisions,key=lambda z:(z['stable_utility'],*z['key'][1:],-V6.CANDIDATES.index(z['candidate'])))
    gate=None
    if chosen['candidate'] in GATE_KINDS:
        gate=fit_gate(chosen['candidate'],records,ids,frame,probs['ungated_control'].video,20261006+9999)
    members=[];fpref={};vpref={};member_parity={}
    for recipe,weight in V6.MEMBERS:
        fp,vp,state=fit_event_member(recipe,records,ids,ids,20261002+9999,full_fit=True)
        member=export_event_member(state);member=json.loads(json.dumps(member,allow_nan=False));member['weight']=weight;members.append(member)
        fpref[recipe]=fp;vpref[recipe]=vp
        probe={'members':[dict(deepcopy(member),weight=1.)],'decoder':{'threshold':.5},'calibration':None}
        mf=mv=0.
        for i,r in enumerate(records):
            p=predict_record(probe,r);f=float(np.max(np.abs(p['frame_probabilities']-fp[i])));v=abs(p['video_probability']-vp[i])
            if max(f,v)>=2e-6:raise ValueError('v6 member portable parity differs')
            mf=max(mf,f);mv=max(mv,v)
        member_parity[recipe]={'frame':mf,'video':mv}
    ref_frame={i:sum(w*fpref[r][i] for r,w in V6.MEMBERS).astype(np.float32) for i in ids}
    ref_video={i:float(sum(w*vpref[r][i] for r,w in V6.MEMBERS)) for i in ids}
    if gate is not None:ref_video={i:apply_gate(gate,ref_frame[i],ref_video[i],records[i]) for i in ids}
    ref=Probabilities(ref_frame,ref_video,None);intervals=V6.decode_predictions(records,ids,ref,chosen['config'])
    bundle={'schema_version':SCHEMA,'recipe':'ensemble','members':members,'decoder':chosen['config'],'calibration':None,'video_gate':gate,
            'raw_feature_names':{k:records[0][k+'_names'] for k in ['motion','corrected','local']},'feature_profiles':profiles,
            'training_video_sha256':[r['sha256'] for r in records],'training_video_names':[r['name'] for r in records],
            'training_role':'all inspected77 annotation rows/76 content; not blind; content conflict retained',
            'model_note':'Offline review; uncalibrated evidence, not physical truth or calibrated risk probability.',
            'evaluation_note':'Same-data iterative nested development scores; fullfit/inner averages are not generalization.',
            'experiment_id':'v6','protocol_sha256':digest(V6.OUT/'protocol.json'),'comparison_report_sha256':digest(V6.OUT/'report.json'),
            'diagnostic_only':not accepted,'default_deployment_acceptance':'accepted_statistically_pending_product' if accepted else 'rejected',
            'statistical_promotion_checks':report['statistical_promotion_checks'],
            'deployment_selection':{'role':'inner_aggregate_only_optimistic_not_validation','candidate':chosen['candidate'],'uses_outer_OOF_probabilities':False},
            'meta_training_note':'Final gate fits inner OOF average, final base uses all77; gate/base training-size shift remains a limitation.'}
    mf=mv=0.
    for i,r in enumerate(records):
        p=predict_record(bundle,r);f=float(np.max(np.abs(p['frame_probabilities']-ref.frame[i])));v=abs(p['video_probability']-ref.video[i])
        if max(f,v)>=2e-6 or p['intervals']!=intervals[i]:raise ValueError('v6 full portable parity differs')
        mf=max(mf,f);mv=max(mv,v)
    bundle['parity']={'rows':len(records),'max_frame_probability_error':mf,'max_video_probability_error':mv,'all_decoded_intervals_equal':True}
    V6.check_receipt();V6.write_new(output/'locator_bundle.json',bundle)
    source_names=['export_video_gate_locator_v6.py','optimized_detector.py','optimized_video_gate_v6.py','optimized_event_training_v5.py','optimized_compact_features_v4.py','optimized_compact_model_v4.py','optimized_duration_decoder_v4.py']
    for n in source_names:
        target=output/'sources'/n;target.parent.mkdir(exist_ok=True)
        with target.open('xb') as f:f.write((ROOT/'code'/n).read_bytes())
    export_report={'status':'complete','role':'inner_aggregate_export_not_validation','diagnostic_only':not accepted,'default_promoted':False,
                   'chosen':chosen,'all_decisions':decisions,'aggregate_inner_counts':[len(ff[i]) for i in ids],
                   'fullfit_training_receipt_sha256':receipt['receipt_sha256'],'before_after_inputs_sources_equal':True,
                   'member_portable_parity':member_parity,'bundle_portable_parity':bundle['parity'],'bundle_sha256':digest(output/'locator_bundle.json'),
                   'source_sha256':{n:digest(ROOT/'code'/n) for n in source_names},'elapsed_seconds':time.perf_counter()-start}
    V6.write_new(output/'report.json',export_report)
    print(json.dumps({'status':'complete','diagnostic_only':not accepted,'chosen':chosen['candidate'],'config':chosen['config'],'parity':bundle['parity']}),flush=True)

if __name__=='__main__':main()
