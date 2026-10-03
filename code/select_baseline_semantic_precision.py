"""Selection-reader precision repair; no changes to frozen fits or outer scores.

Original final TCN video readout was quantized to float32. Restore native
float64 baseline probability storage; keep frame values, models and all rules.
"""
from __future__ import annotations
import argparse
import gzip
import json
import pickle
from pathlib import Path
import numpy as np
import run_baseline_semantic_readout as R

OUT=R.RUN/'video-precision-repair'


def original_positive(model,values):
    matches=np.flatnonzero(model.classes_==1)
    if not len(matches):return np.zeros(len(values),np.float64)
    # Baseline predict_proba is float64. Do not cast video scores to float32.
    return model.predict_proba(values)[:,int(matches[0])]


def freeze():
    p=R.check_protocol(inputs=True)
    if OUT.exists():raise FileExistsError('refuse precision-repair namespace overwrite')
    OUT.mkdir();(OUT/'sources').mkdir()
    for name in ('select_baseline_semantic_precision.py','run_baseline_semantic_readout.py'):
        with (OUT/'sources'/name).open('xb') as f:f.write((R.ROOT/'code'/name).read_bytes())
    receipt={'schema_version':'semantic-final-tcn-video-precision-reader-v1',
        'parent_protocol_receipt':p['receipt_sha256'],
        'original_runner_sha256':R.B.digest(R.ROOT/'code/run_baseline_semantic_readout.py'),
        'adapter_source_sha256':R.B.digest(Path(__file__)),
        'parent_report_sha256':R.B.digest(R.RUN/'report.json'),
        'original_deployment_selection_sha256':R.B.digest(R.RUN/'deployment_selection.json'),
        'scope':'final inner TCN video only: native predict_proba float64, instead of float32 cast',
        'unchanged':['46 fitted semantic members','23 fit partitions and seeds','PCA states','all frame probabilities',
            '5 outer choices/predictions/metrics/guards','original default decoder','two-mode selection rule','default runtime'],
        'additional_member_fits':0,'additional_feature_extractions':0,'old_outputs_preserved':True}
    receipt['receipt_sha256']=R.B.object_hash(receipt)
    R.B.write_new(OUT/'selection_reader_receipt.json',receipt)
    print(json.dumps({'status':'precision_reader_frozen','receipt':receipt['receipt_sha256']}),flush=True)


def check_receipt():
    p=R.check_protocol();receipt=json.loads((OUT/'selection_reader_receipt.json').read_text('utf-8'))
    if receipt['receipt_sha256']!=R.B.object_hash({k:v for k,v in receipt.items() if k!='receipt_sha256'}):
        raise ValueError('precision receipt differs')
    for path,key in ((Path(__file__),'adapter_source_sha256'),(R.ROOT/'code/run_baseline_semantic_readout.py','original_runner_sha256'),
                     (R.RUN/'report.json','parent_report_sha256'),(R.RUN/'deployment_selection.json','original_deployment_selection_sha256')):
        if R.B.digest(path)!=receipt[key]:raise ValueError('precision source/report drift')
    if p['receipt_sha256']!=receipt['parent_protocol_receipt']:raise ValueError('parent protocol drift')
    return p,receipt


def select():
    p,receipt=check_receipt();_,records,_,_=R.feature_records();ids=list(range(len(records)))
    original=json.loads((R.RUN/'deployment_selection.json').read_text('utf-8'))
    before={m:{int(i):[tuple(x) for x in value] for i,value in original['predictions'][m].items()} for m in R.MODES}
    final={m:R.Probabilities({},{}) for m in R.MODES};partitions=[];changes=0;max_tcn=0.;max_fused=0.;control_error=0.
    score_arrays={};model_bindings=[]
    for inner in range(3):
        name=f'final_inner_{inner}'
        control,tcn,meta=R.final_control_and_tcn(name,records)
        path=R.ROOT/meta['model_state_path']
        if R.B.digest(path)!=meta['model_state_sha256']:raise ValueError('trusted state bytes changed')
        with gzip.open(path,'rb') as f:states=pickle.load(f)
        corrected=R.Probabilities(tcn.frame,{});replayed={}
        for i in meta['predict']:
            native_v={}
            for recipe,_ in R.B.MEMBERS:
                member=states[recipe];x,v,names=R.feature_view(recipe,records[i],member['transform']['pca'])
                if names!=member['transform']['frame_feature_names']:raise ValueError('baseline state view order differs')
                native_v[recipe]=float(original_positive(member['video_model'],v[None])[0])
            corrected.video[i]=native_v['rgb_motion_tcn']
            native_blend=sum(weight*native_v[recipe] for recipe,weight in R.B.MEMBERS)
            control_error=max(control_error,abs(native_blend-control.video[i]))
            difference=abs(tcn.video[i]-corrected.video[i]);max_tcn=max(max_tcn,difference);changes+=int(difference!=0)
        if control_error>2e-12:raise ValueError('native baseline video replay differs from original control cache')
        semantic,newmeta=R.load_job(name,records)
        if any(newmeta[key]!=meta[key] for key in ('fit','predict','seed')):raise ValueError('final splits differ')
        old_enabled=R.fuse(semantic,tcn,meta['predict'],records)
        enabled=R.fuse(semantic,corrected,meta['predict'],records)
        for i in meta['predict']:
            if not np.array_equal(old_enabled.frame[i],enabled.frame[i]):raise ValueError('precision repair changed frame')
            max_fused=max(max_fused,abs(old_enabled.video[i]-enabled.video[i]))
            score_arrays[f'tcn_video_original_precision_{i}']=np.asarray(corrected.video[i],np.float64)
            score_arrays[f'enabled_video_original_precision_{i}']=np.asarray(enabled.video[i],np.float64)
        for dest,src in ((final[R.MODES[0]],control),(final[R.MODES[1]],enabled)):
            if set(dest.frame)&set(src.frame):raise ValueError('duplicate final rows')
            dest.frame.update(src.frame);dest.video.update(src.video)
        partitions.append({'fit':meta['fit'],'validation':meta['predict'],'seed':meta['seed']})
        model_bindings.append({'job':name,'base_model_state_sha256':meta['model_state_sha256'],
                               'semantic_fit_receipt':newmeta['receipt_sha256']})
    decoded={m:R.decode_scores(records,ids,final[m],p['default_decoder']) for m in R.MODES}
    changed_intervals={m:[i for i in ids if decoded[m][i]!=before[m][i]] for m in R.MODES}
    choice=R.choose_mode(records,ids,decoded,20273111)
    old_choice=original['chosen']['mode']
    row={'status':'complete','role':'final_inner_OOF_deployment_selection_optimistic_not_validation',
        'parent_protocol_receipt':p['receipt_sha256'],'precision_reader_receipt':receipt['receipt_sha256'],
        'inner_partitions':partitions,'chosen':choice,'predictions':{m:{str(i):decoded[m][i] for i in ids} for m in R.MODES},
        'model_bindings':model_bindings,'evidence':{'TCN_video_changed_rows':changes,'TCN_video_max_difference':max_tcn,
            'fused_video_max_difference':max_fused,'native_control_video_replay_max_error':control_error,
            'changed_intervals':changed_intervals,'choice_changed':choice['mode']!=old_choice,
            'frames_identical':True,'all_outer_results_preserved':True,'additional_member_fits':0},
        'statistical_promotion_passed':original['statistical_promotion_passed'],'default_promoted':False,'outer_best_not_used':True}
    if not (OUT/'final_video_probabilities.npz').exists():
        with (OUT/'final_video_probabilities.npz').open('xb') as f:np.savez_compressed(f,**score_arrays)
    else:raise FileExistsError('refuse corrected final probabilities overwrite')
    row['corrected_score_sha256']=R.B.digest(OUT/'final_video_probabilities.npz')
    row['receipt_sha256']=R.B.object_hash(row);R.B.write_new(OUT/'deployment_selection.json',row)
    if not row['statistical_promotion_passed'] or choice['mode']==R.MODES[0]:
        R.B.write_new(OUT/'final_fit_skipped.json',{'status':'skipped','reason':'statistical guards failed' if not row['statistical_promotion_passed'] else 'inner OOF selected baseline',
            'planned_unused_full_member_fits':2,'default_promoted':False,'export_created':False,'precision_reader_receipt':receipt['receipt_sha256']})
    check_receipt();print(json.dumps({'status':'complete','mode':choice['mode'],'evidence':row['evidence']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','select'));args=parser.parse_args()
    if args.action=='freeze':freeze()
    else:select()
