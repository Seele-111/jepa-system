#!/usr/bin/env python3
"""Inner-selected full-fit diagnostic export. Never installs or promotes a model."""
from __future__ import annotations
import argparse, importlib, json, time
import numpy as np
from optimized_grouped_training import ROOT, load_grouped, digest
from optimized_feature_blocks_v10 import fit_block_member, export_block_member, predict_block_member
from optimized_compact_model_v4 import fit_compact_member, export_compact_member, predict_compact_member
from optimized_proposal_review_v11 import verify_proposals
from optimized_training_provenance import collect_new_fit_inputs
from optimized_detector import predict_record, required_channels
from optimized_locator import SCHEMA
from run_compact_experiment_v4 import write_new, object_hash, decode_predictions
from run_optimization_selection import Probabilities
import run_feature_blocks_v10 as V10


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',choices=['v10','v11'],required=True)
    parser.add_argument('--allow-failed-experiment',action='store_true');args=parser.parse_args()
    exp=importlib.import_module('run_feature_blocks_v10' if args.experiment=='v10' else 'run_proposal_review_v11')
    start=time.perf_counter();receipt=exp.check_receipt();_,records,profiles=load_grouped();ids=list(range(len(records)))
    report=json.loads((exp.OUT/'report.json').read_text('utf-8'));selection=json.loads((exp.OUT/'deployment_selection.json').read_text('utf-8'))
    if selection['uses_outer_probabilities'] or selection['uses_outer_metrics_for_selection'] or selection['coverage_per_row']!=1:
        raise ValueError('unsafe deployment selection evidence')
    passed=bool(report['statistical_promotion_passed'])
    if not passed and not args.allow_failed_experiment:raise ValueError('explicit diagnostic-only export required')
    out=exp.OUT/('deployment_candidate' if passed else 'deployment_diagnostic');out.mkdir(exist_ok=False)
    source=collect_new_fit_inputs()
    source_names=['export_diagnostic_v10_v11.py','optimized_detector.py','optimized_feature_blocks_v10.py','optimized_compact_features_v4.py','optimized_compact_model_v4.py','optimized_proposal_review_v11.py','optimized_event_training_v5.py','optimized_locator.py']
    fullfit_receipt={'role':'before_fit_receipt_all_development_content_not_validation','input_hashes':dict(source.input_hashes),
        'inputs_sha256':source.inputs_sha256,'sources':{n:digest(ROOT/'code'/n) for n in source_names},
        'selection_sha256':digest(exp.OUT/'deployment_selection.json'),'training_receipt_sha256':receipt['receipt_sha256']}
    fullfit_receipt['receipt_sha256']=object_hash(fullfit_receipt);write_new(out/'fullfit_receipt.json',fullfit_receipt)
    chosen=selection['chosen'];recipe=chosen['candidate'] if args.experiment=='v10' else V10.RICH
    fp,vp,state=fit_block_member(recipe,records,ids,ids,20261002+5*53+99,full_fit=True)
    member=json.loads(json.dumps(export_block_member(state),allow_nan=False));member['weight']=1.
    config=chosen['config'] if args.experiment=='v10' else chosen['config']['proposal']
    bundle={'schema_version':SCHEMA,'recipe':recipe,'members':[member],'calibration':None,'decoder':config,
        'raw_feature_names':{k:records[0][k+'_names'] for k in ['motion','corrected','local']},'feature_profiles':profiles,
        'training_video_sha256':[r['sha256'] for r in records],'training_video_names':[r['name'] for r in records],
        'training_role':'all77 inspected development annotations/76content; not blind',
        'model_note':'Offline human review; uncalibrated learned evidence, not physical truth or risk probabilities.',
        'evaluation_note':'Nested selection pipeline scores are not unseen performance of this full-fit export.',
        'experiment_id':args.experiment,'diagnostic_only':not passed,
        'default_deployment_acceptance':'accepted_statistically_pending_product' if passed else 'rejected',
        'statistical_promotion_checks':report['statistical_promotion_checks'],
        'protocol_sha256':digest(exp.OUT/'protocol.json'),'comparison_report_sha256':digest(exp.OUT/'report.json'),
        'deployment_selection':{'role':selection['role'],'candidate':chosen['candidate'],'uses_outer_OOF_probabilities':False}}
    base=Probabilities(fp,vp,None);expected=decode_predictions(records,ids,base,config);review_fp=None
    errors={'primary_frame':0.,'primary_video':0.,'reviewer_frame':0.,'reviewer_video':0.}
    if args.experiment=='v11' and chosen['config']['review']['minimum']>0:
        qf,qv,qstate=fit_compact_member(exp.REVIEWER,records,ids,ids,20261002+5*53+99,full_fit=True)
        reviewer=json.loads(json.dumps(export_compact_member(qstate),allow_nan=False));review_fp=qf
        bundle['proposal_verifier']={'schema_version':'proposal-verifier-v11','member':reviewer,'review':chosen['config']['review']}
        for i,r in enumerate(records):
            f,v=predict_compact_member(reviewer,r)
            errors['reviewer_frame']=max(errors['reviewer_frame'],float(np.max(np.abs(f-qf[i]))))
            errors['reviewer_video']=max(errors['reviewer_video'],abs(v-qv[i]))
            expected[i],_=verify_proposals(expected[i],fp[i],qf[i],chosen['config']['review'])
    for i,r in enumerate(records):
        f,v=predict_block_member(member,r)
        errors['primary_frame']=max(errors['primary_frame'],float(np.max(np.abs(f-fp[i]))))
        errors['primary_video']=max(errors['primary_video'],abs(v-vp[i]))
        pred=predict_record(bundle,r)
        if max(float(np.max(np.abs(pred['frame_probabilities']-fp[i]))),abs(pred['video_probability']-vp[i]))>2e-6 or pred['intervals']!=expected[i]:
            raise ValueError('full bundle portable parity differs')
    if max(errors.values())>2e-6:raise ValueError('portable member parity differs')
    after=collect_new_fit_inputs()
    if dict(after.input_hashes)!=fullfit_receipt['input_hashes'] or after.inputs_sha256!=fullfit_receipt['inputs_sha256'] or any(digest(ROOT/'code'/n)!=h for n,h in fullfit_receipt['sources'].items()):
        raise ValueError('fullfit input/source changed')
    exp.check_receipt()
    bundle['parity']={'rows':len(records),'member_max_errors':errors,'all_intervals_equal':True}
    write_new(out/'locator_bundle.json',bundle)
    for n in source_names:
        target=out/'sources'/n;target.parent.mkdir(exist_ok=True)
        with target.open('xb') as f:f.write((ROOT/'code'/n).read_bytes())
    result={'status':'complete','role':'diagnostic_fullfit_export_not_validation','diagnostic_only':not passed,
        'default_promoted':False,'candidate':chosen['candidate'],'config':chosen['config'],
        'required_channels':sorted(required_channels(bundle)),'portable_parity':bundle['parity'],
        'bundle_sha256':digest(out/'locator_bundle.json'),'default_models_unchanged':True,
        'fullfit_receipt_sha256':fullfit_receipt['receipt_sha256'],'elapsed_seconds':time.perf_counter()-start}
    write_new(out/'report.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
