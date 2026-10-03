#!/usr/bin/env python3
"""Independent statistical replay for v13 using the independently reviewed v10 auditor.

No estimator fitting. Spatial/compact transforms and decoder are audited dependencies;
selection, weighting, matching, bootstrap, ranking and metrics are independently replayed.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
import audit_feature_blocks_v10 as A

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('output/algorithm-opt-v13-spatial-jepa')
RECEIPT_PIN='043f2251dfaa802ed41eb865520c1b249a60aec0c34162806b3d7aa1c874ba26'
RECIPE='spatial_jepa_corrected_motion_et';CONTROL='blocks_corrected_motion_et'


def verify_spatial_cache(e,scope,records,outer,receipt,*,final=False):
    path=OUT/('final_selection' if final else 'training')/f'{scope}_{RECIPE}.npz'
    row=e.json(path.with_suffix('.json'))
    A.require(row.get('signature')==A.value_hash({k:v for k,v in row.items() if k!='signature'}),'spatial metadata signature differs')
    ids=list(range(len(records)));val=[] if final else outer[scope];train=[i for i in ids if i not in set(val)]
    parts=A.expected_partitions(records,train,scope)
    for key,value in [('scope',scope),('final_selection',final),('recipe',RECIPE),('train',train),('validation',val),
        ('inner_partitions',parts),('training_receipt_sha256',receipt['receipt_sha256']),('outer_seed',None if final else 20261002+scope*53+99)]:
        A.compare(row.get(key),value,'spatial/'+str(scope)+'/'+key)
    expected=parts+([] if final else [{'fit':train,'validation':val}])
    A.require(len(row['fit_evidence'])==len(expected),'spatial fit evidence count differs')
    from optimized_spatial_jepa_v13 import feature_view_spatial
    for k,(proof,part) in enumerate(zip(row['fit_evidence'],expected)):
        A.check_partition(records,part['fit'],part['validation'],train if k<3 else ids,'spatial/'+str(scope)+'/'+str(k))
        A.compare(proof['partition'],k if k<3 else 'outer','spatial fit partition ordinal')
        A.compare(proof['fit_content_sha256'],sorted({records[i]['sha256'] for i in part['fit']}),'spatial fit content')
        _,_,names=feature_view_spatial(RECIPE,records[min(part['fit'])],None)
        transform={'feature_view':'spatial-jepa-v13','pca':None,'frame_feature_names':names}
        A.compare(proof['transform_sha256'],A.value_hash(transform),'spatial label-free transform digest')
    arrays=A.read_archive(e.read(path,row['npz_sha256']),path.as_posix())
    keys={f'{kind}_{head}_{i}' for kind,sub in [('inner',train),('outer',val)] for i in sub for head in ['frame','video']}
    A.require(set(arrays)==keys,'spatial probability coverage differs')
    for kind,sub in [('inner',train),('outer',val)]:
        for i in sub:
            for head,shape in [('frame',(records[i]['frames'],)),('video',())]:
                x=arrays[f'{kind}_{head}_{i}'];A.require(x.shape==shape and np.isfinite(x).all() and np.all((x>=0)&(x<=1)),'spatial tensor shape/range differs')
    def read(kind,sub):return A.Probabilities({i:arrays[f'{kind}_frame_{i}'] for i in sub},{i:float(arrays[f'{kind}_video_{i}']) for i in sub})
    return read('inner',train),read('outer',val),row


def rank(rows):
    A.compare([r['candidate'] for r in rows],[RECIPE,'control'],'spatial candidate order')
    return max(rows,key=lambda z:(z['stable_utility'],*z['key'][1:],-[RECIPE,'control'].index(z['candidate'])))


def run_audit():
    start=time.perf_counter();e=A.Evidence(ROOT)
    receipt=e.json(OUT/'training_receipt.json');A.require(receipt.get('receipt_sha256')==RECEIPT_PIN and A.value_hash({k:v for k,v in receipt.items() if k!='receipt_sha256'})==RECEIPT_PIN,'spatial frozen receipt differs')
    parent,baseline=A.verify_receipts(e)
    A.compare(receipt['parents']['v10'],parent['receipt_sha256'],'spatial control parent')
    A.compare(receipt['parents']['v8'],parent['parent_receipt_sha256'],'spatial rich parent')
    A.compare(receipt['input_hashes'],parent['input_hashes'],'spatial original labels/FPS/feature binding')
    A.compare(receipt['inputs_sha256'],parent['inputs_sha256'],'spatial input inventory digest')
    for name,h in receipt['sources'].items():
        e.read('code/'+name,h);e.read(OUT/'sources'/name,h)
    for section in ['control_artifacts']:
        for path,h in receipt[section].items():e.read(path,h)
    for path,h in receipt['raw_spatial']['files'].items():e.read(path,h)
    protocol=e.json(OUT/'protocol.json',receipt['protocol_sha256']);report=e.json(OUT/'report.json');deployment=e.json(OUT/'deployment_selection.json')
    A.compare(protocol['promotion_guards'],A.GUARD_SPEC,'spatial guard spec')
    A.compare(report['protocol_sha256'],receipt['protocol_sha256'],'spatial report protocol')
    A.compare(report['training_receipt_sha256'],RECEIPT_PIN,'spatial report receipt')
    A.compare(report['grouped_v1_baseline'],baseline,'spatial baseline lineage')
    records,outer,aliases=A.load_records(e)
    from optimized_spatial_jepa_v13 import spatial_from_raw,RAW_ROOT
    rm=Path('output/algorithm-opt-2026-10-02/corrected_jepa_features_v2');raw_manifest=e.json(rm/'manifest.json',receipt['raw_spatial']['manifest_sha256'])
    A.require(raw_manifest.get('status')=='complete' and raw_manifest.get('label_free') is True and len(raw_manifest['videos'])==len(records),'raw spatial manifest invalid')
    for i,(r,entry) in enumerate(zip(records,raw_manifest['videos'])):
        for key,value in [('name',r['name']),('source_sha256',r['sha256']),('fps',r['fps']),('frames',r['frames'])]:A.compare(entry[key],value,'raw spatial row identity')
        relative=(rm/entry['raw_directory']/'signals.npz').as_posix();data=e.read(relative,receipt['raw_spatial']['files'][relative])
        values,mask,names=spatial_from_raw(ROOT/relative,r['frames'],r['fps'])
        r.update(spatial=values,spatial_support=mask,spatial_names=names)
    decoders=A.decoder_functions();transforms=A.TransformEvidence(records);selected={};fixed={name:{} for name in [RECIPE,'control']};decisions=[];fit_proofs=[]
    A.require(len(report['folds'])==5,'spatial outer fold count differs')
    for scope,val in enumerate(outer):
        ids=[i for i in range(len(records)) if i not in set(val)]
        ip,op,meta=verify_spatial_cache(e,scope,records,outer,receipt);ci,co,cmeta=A.load_cache(e,CONTROL,scope,records,outer,transforms)
        A.compare(meta['inner_partitions'],cmeta['inner_partitions'],'spatial/control partition parity')
        rows=[]
        for name,p in [(RECIPE,ip),('control',ci)]:
            choice=A.independent_choice(records,ids,p,20261002+scope,decoders);choice['candidate']=name;rows.append(choice)
        chosen=rank(rows);saved=report['folds'][scope]
        A.compare(saved['primary'],chosen,'spatial primary selection')
        A.compare(saved['all_inner_selections'],rows,'spatial all inner selections')
        for key,value in [('fold',scope),('train_indices',ids),('validation_indices',val),('inner_partitions',meta['inner_partitions'])]:A.compare(saved[key],value,'spatial report partition')
        pred=A.decode(records,val,op if chosen['candidate']==RECIPE else co,chosen['config'],decoders)
        A.compare(saved['primary_predictions'],{str(i):pred[i] for i in val},'spatial primary predictions')
        A.require(not set(selected)&set(pred),'spatial outer OOF repeated');selected.update(pred)
        for choice in rows:fixed[choice['candidate']].update(A.decode(records,val,op if choice['candidate']==RECIPE else co,choice['config'],decoders))
        decisions.append({'fold':scope,'candidate':chosen['candidate'],'config':chosen['config']});fit_proofs.extend(meta['fit_evidence'])
    primary=report['primary_v13_nested'];A.compare(primary['metrics'],A.grouped_metrics(records,selected),'spatial primary metrics');A.compare(primary['predictions'],A.prediction_rows(records,selected),'spatial prediction serialization');A.compare(primary['summary'],A.prediction_summary(records,selected),'spatial summary')
    for name,pred in fixed.items():
        saved=report['fixed_candidate_diagnostics_not_for_promotion'][name]
        A.compare(saved['metrics'],A.grouped_metrics(records,pred),'spatial fixed metrics');A.compare(saved['predictions'],A.prediction_rows(records,pred),'spatial fixed predictions')
    bp=A.read_prediction_rows(baseline['predictions'],records,'spatial baseline');bm=A.grouped_metrics(records,bp);guard=A.promotion_guards(A.grouped_metrics(records,selected),bm)
    A.compare(report['statistical_promotion_checks'],guard,'spatial guards');A.compare(report['statistical_promotion_passed'],all(guard.values()),'spatial promotion')
    A.compare(report['paired_content_bootstrap'],A.paired_uncertainty(records,bp,selected),'spatial uncertainty')
    ids=list(range(len(records)));ip,_,meta=verify_spatial_cache(e,5,records,outer,receipt,final=True);ci,_,cmeta=A.load_cache(e,CONTROL,5,records,outer,transforms,final=True);rows=[]
    A.compare(meta['inner_partitions'],cmeta['inner_partitions'],'spatial final/control partitions')
    for name,p in [(RECIPE,ip),('control',ci)]:choice=A.independent_choice(records,ids,p,20261007,decoders);choice['candidate']=name;rows.append(choice)
    chosen=rank(rows);A.compare(deployment['chosen'],chosen,'spatial final selection');A.compare(deployment['all_inner_selections'],rows,'spatial final rows')
    A.compare(deployment['inner_partitions'],meta['inner_partitions'],'spatial final parts')
    for key,value in [('uses_outer_probabilities',False),('uses_outer_metrics_for_selection',False),('coverage_per_row',1),('training_receipt_sha256',RECEIPT_PIN)]:A.compare(deployment[key],value,'spatial final evidence')
    fit_proofs.extend(meta['fit_evidence']);e.recheck()
    return {'status':'passed','scope':'independent_v13_statistical_replay_and_structural_cache_audit',
        'deployment_approved':False,'receipt_sha256':RECEIPT_PIN,'rows':len(records),'aliases':aliases,
        'outer_decisions':decisions,'final_choice':{'candidate':chosen['candidate'],'config':chosen['config']},
        'fit_proofs':len(fit_proofs),'control_transform_proofs':len(transforms.proofs),'primary_metrics':primary['metrics'],
        'statistical_guards':guard,'differing_consumed_files':0,'consumed_artifact_sha256':dict(sorted(e.hashes.items())),
        'auditor_source_sha256':A.sha(Path(__file__).read_bytes()),'statistical_auditor_source_sha256':A.sha((ROOT/'code/audit_feature_blocks_v10.py').read_bytes()),
        'elapsed_seconds':time.perf_counter()-start,
        'limitations':A.LIMITATIONS+['Spatial transform is a source-reviewed audited dependency, not a second independent feature implementation.',
            'No estimator states were saved for per-fold model replay; fit-content and no-PCA transform digests are structural evidence.',
            'V13 raw heatmap hashing binds present bytes; it does not recover missing historical raw extraction/source authentication.',
            'No V13 live worker spatial extractor or deployment bundle is installed; failed experimental recipe is not product-supported.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('refuse audit overwrite')
    result=run_audit()
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({'status':result['status'],'rows':result['rows'],'fit_proofs':result['fit_proofs'],'deployment_approved':False}))
if __name__=='__main__':main()
