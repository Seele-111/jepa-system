#!/usr/bin/env python3
"""Nested held-out-video algorithm optimization; sklearn is used only for fitting."""
from __future__ import annotations
import argparse, hashlib, itertools, json, time
from pathlib import Path
from datetime import datetime
import numpy as np
from optimized_locator import augment_signals, video_features, decode, metrics, objective, export_model, portable_predict, SCHEMA
from build_optimization_dataset import make_folds
from optimized_feature_view import feature_view

ROOT=Path(__file__).resolve().parents[1]


def _json_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def _read_optional_features(root, folder):
    manifest_path=root/folder/'manifest.json'
    if not manifest_path.is_file():return {}
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    records=manifest.get('videos',manifest.get('records',manifest.get('results',[])))
    if isinstance(records,dict):records=[dict(v,name=k) for k,v in records.items()]
    output={}
    for record in records:
        if record.get('status','ok') not in ['ok','done','success','complete']:continue
        name=record.get('name',record.get('video_name'))
        file=record.get('file',record.get('npz_path'))
        if not name or not file:raise ValueError('feature record lacks name/file: '+folder)
        if name in output:raise ValueError('duplicate feature name: '+name)
        path=manifest_path.parent/Path(file)
        if not path.is_file():raise FileNotFoundError(path)
        expected_hash=record.get('feature_sha256')
        if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hash:
            raise ValueError('feature hash mismatch: '+name)
        with np.load(path,allow_pickle=False) as archive:
            keys=[k for k in ['signals','features'] if k in archive.files]
            if len(keys)!=1:raise ValueError('ambiguous/missing feature array: '+name)
            values=np.asarray(archive[keys[0]],dtype=np.float32)
            if values.ndim!=2 or not len(values) or not np.isfinite(values).all():
                raise ValueError('invalid feature array: '+name)
            output[name]=values
    return output


def experiment_signature(data_root, recipes, corrected_folder='corrected_jepa_features_v2'):
    paths=[Path(__file__),Path(__file__).with_name('optimized_locator.py'),
           Path(__file__).with_name('build_optimization_dataset.py'),Path(__file__).with_name('optimized_feature_view.py'),
           data_root/'dataset_manifest.json',data_root/'dataset.npz']
    if any(r.endswith('_tcn') for r in recipes):paths.append(Path(__file__).with_name('optimized_temporal_head.py'))
    if any('boundary' in r for r in recipes):paths.append(Path(__file__).with_name('optimized_boundary_head.py'))
    folders=[]
    if any('motion' in r or 'hybrid' in r for r in recipes):folders.append('motion_features')
    if any(r.startswith('rgb_') for r in recipes):folders.append('rgb_features')
    if any('corrected' in r for r in recipes):folders.append(corrected_folder)
    for folder in folders:
        manifest_path=data_root/folder/'manifest.json'
        if not manifest_path.is_file():raise FileNotFoundError(manifest_path)
        paths.append(manifest_path)
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        for record in manifest.get('videos',[]):
            if record.get('status')!='ok':continue
            file=record.get('file',record.get('npz_path'))
            if file:paths.append(manifest_path.parent/file)
    sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    signature=hashlib.sha256(json.dumps({'sources':sources,'recipes':recipes},sort_keys=True).encode()).hexdigest()
    return signature,sources


def make_view(recipe, records, train_indices, seed):
    """Every learned transformation is fit on training videos only."""
    pca_state=None
    if recipe.startswith('rgb_'):
        from sklearn.decomposition import PCA
        pool=np.concatenate([records[i]['rgb'] for i in train_indices])
        pca=PCA(n_components=min(16,len(pool),pool.shape[1]),svd_solver='randomized',random_state=seed).fit(pool)
        pca_state={'mean':pca.mean_.tolist(),'components':pca.components_.tolist()}
    frame_values=[];video_values=[];frame_names=None
    for record in records:
        frames,video,names=feature_view(recipe,record,pca_state)
        frame_values.append(frames);video_values.append(video)
        if frame_names is None:frame_names=names
        elif frame_names!=names:raise ValueError('feature order differs between videos')
    transform={'recipe':recipe,'frame_feature_names':frame_names,'pca':pca_state,
               'feature_view':'shared-label-free-v1'}
    return frame_values,np.stack(video_values),transform


def fit_predict(recipe,records,train_indices,predict_indices,seed):
    from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
    frame_values,video_values,transform=make_view(recipe,records,train_indices,seed)
    x=np.concatenate([frame_values[i] for i in train_indices]);y=np.concatenate([records[i]['labels'] for i in train_indices])
    weights=np.concatenate([np.full(len(records[i]['labels']),1/len(records[i]['labels']),np.float64) for i in train_indices])
    positive=weights[y>0].sum();negative=weights[y==0].sum()
    weights=np.where(y>0,weights*negative/max(1e-8,positive),weights)
    weights*=len(weights)/weights.sum()
    if recipe.endswith('_gb'):
        model=GradientBoostingClassifier(n_estimators=120,learning_rate=.05,max_depth=2,min_samples_leaf=8,
                                        subsample=.85,random_state=seed)
    elif recipe.endswith('_et'):
        model=ExtraTreesClassifier(n_estimators=192,max_depth=9,min_samples_leaf=8,max_features=.6,n_jobs=4,random_state=seed)
    else:
        model=RandomForestClassifier(n_estimators=160,max_depth=8,min_samples_leaf=10,max_features=.5,n_jobs=4,random_state=seed)
    if recipe.endswith('_tcn'):
        from optimized_temporal_head import train_temporal
        model=train_temporal([frame_values[i] for i in train_indices],
                             [records[i]['labels'] for i in train_indices],
                             [records[i]['fps'] for i in train_indices],seed,steps=400)
    else:model.fit(x,y,sample_weight=weights)
    video_y=np.asarray([bool(np.any(r['labels'])) for r in records],dtype=np.int64)
    video_model=ExtraTreesClassifier(n_estimators=128,max_depth=4,min_samples_leaf=2,max_features=.75,
                                    class_weight='balanced',n_jobs=4,random_state=seed+1100)
    video_model.fit(video_values[train_indices],video_y[train_indices])
    if recipe.endswith('_tcn'):
        from optimized_temporal_head import predict_temporal
        frame_out={i:predict_temporal(model,frame_values[i],records[i]['fps']) for i in predict_indices}
    else:frame_out={i:model.predict_proba(frame_values[i])[:,1].astype(np.float32) for i in predict_indices}
    vp=video_model.predict_proba(video_values[predict_indices])[:,1]
    video_out={i:float(p) for i,p in zip(predict_indices,vp)}
    extra={'boundary_models':None,'boundary_probabilities':None}
    if 'boundary' in recipe:
        from optimized_boundary_head import train_boundary_heads,predict_boundary_heads
        extra['boundary_models']=train_boundary_heads([frame_values[i] for i in train_indices],
            [records[i]['labels'] for i in train_indices],[records[i]['fps'] for i in train_indices],seed+1700)
        extra['boundary_probabilities']={i:predict_boundary_heads(extra['boundary_models'],frame_values[i]) for i in predict_indices}
    return frame_out,video_out,(model,video_model,transform,frame_values,video_values,extra)


def decoder_grid():
    for threshold,ratio,smooth,gap,minimum in itertools.product([.3,.4,.5,.6,.7,.8],[1.0,.7],[0,.15],[0,.12],[.08,.20]):
        yield {'threshold':threshold,'low_ratio':ratio,'smooth_seconds':smooth,'gap_seconds':gap,'min_seconds':minimum}


def choose_decoder(records,indices,frame_probs,video_probs,boundary_probs=None):
    labels=[records[i]['labels'] for i in indices]
    best=None;best_key=None
    for config in decoder_grid():
        for strength in [0.0,0.5]:
            adjusted={**config,'video_strength':strength}
            base=[decode(frame_probs[i],records[i]['fps'],video_probs[i],adjusted) for i in indices]
            for seconds in ([0,.2,.4] if boundary_probs is not None else [0]):
                refined=base
                if seconds:
                    from optimized_boundary_head import refine_intervals
                    refined=[refine_intervals(p,*boundary_probs[i],records[i]['fps'],{'boundary_seconds':seconds}) for i,p in zip(indices,base)]
                for vt in [0,.35,.5,.65,.75]:
                    preds=[p if video_probs[i]>=vt else [] for i,p in zip(indices,refined)]
                    result=metrics(preds,labels)
                    key=(objective(result),result['iou_0.5']['f1'],result['iou_0.3']['precision'],-result['predicted_segments'])
                    if best_key is None or key>best_key:best_key=key;best=({**adjusted,'video_threshold':vt,'boundary_seconds':seconds},result)
    return best


def evaluate_probabilities(records,indices,fp,vp,config,boundary_probs=None):
    predictions={i:decode(fp[i],records[i]['fps'],vp[i],config) for i in indices}
    if config.get('boundary_seconds',0):
        if boundary_probs is None:raise ValueError('missing boundary evidence')
        from optimized_boundary_head import refine_intervals
        predictions={i:refine_intervals(predictions[i],*boundary_probs[i],records[i]['fps'],config) for i in indices}
    return predictions,metrics(list(predictions.values()),[records[i]['labels'] for i in indices])


def _group_metrics(records,predictions):
    result={}
    for name,condition in [('all',lambda r:True),('normal',lambda r:r['event_count']==0),
                           ('single_event',lambda r:r['event_count']==1),('multi_event',lambda r:r['event_count']>1)]:
        indices=[i for i,r in enumerate(records) if condition(r)]
        result[name]=metrics([predictions[i] for i in indices],[records[i]['labels'] for i in indices])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=ROOT/'output'/'algorithm-opt-2026-10-02')
    p.add_argument('--output',type=Path);p.add_argument('--recipes',default='jepa_rf,motion_rank_rf')
    p.add_argument('--corrected-folder',default='corrected_jepa_features_v2');
    p.add_argument('--outer-folds',default='all');p.add_argument('--deployment-fit',action='store_true')
    args=p.parse_args();output=args.output or args.data_root/'nested_cv';output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.data_root/'dataset_manifest.json').read_text(encoding='utf-8'))
    archive=np.load(args.data_root/'dataset.npz',allow_pickle=False)
    enhanced=_read_optional_features(args.data_root,'motion_features');rgb=_read_optional_features(args.data_root,'rgb_features')
    corrected=_read_optional_features(args.data_root,args.corrected_folder)
    corrected_names_path=args.data_root/args.corrected_folder/'feature_names.json'
    corrected_names=json.loads(corrected_names_path.read_text(encoding='utf-8')) if corrected_names_path.is_file() else None
    motion_names_path=args.data_root/'motion_features'/'feature_names.json'
    motion_names=json.loads(motion_names_path.read_text(encoding='utf-8')) if motion_names_path.is_file() else None
    records=[]
    for i,row in enumerate(manifest['rows']):
        r={**row,'jepa':archive[f'jepa_{i}'],'labels':archive[f'labels_{i}'],'rank_motion':archive[f'rank_motion_{i}'],
           'jepa_names':manifest['feature_names']['jepa']}
        if row['name'] in enhanced:
            r['motion']=enhanced[row['name']];r['motion_names']=motion_names or [f'motion_{j}' for j in range(r['motion'].shape[1])]
        if row['name'] in rgb:r['rgb']=rgb[row['name']]
        if row['name'] in corrected:r['corrected']=corrected[row['name']];r['corrected_names']=corrected_names
        for key in ['jepa','rank_motion','motion','rgb','corrected']:
            if key in r and (len(r[key])!=len(r['labels']) or not np.isfinite(r[key]).all()):raise ValueError('misaligned '+key+' '+r['name'])
        records.append(r)
    recipes=args.recipes.split(',');N=len(records);all_indices=list(range(N));outer=manifest['outer_folds']
    chosen_outer=list(range(len(outer))) if args.outer_folds=='all' else [int(s) for s in args.outer_folds.split(',')]
    report={'schema_version':'algorithm-opt-nested-cv-v1','protocol':manifest['protocol'],'role':manifest['role'],
            'recipes':recipes,'folds':[],'elapsed_seconds':0,'dataset_manifest_sha256':hashlib.sha256((args.data_root/'dataset_manifest.json').read_bytes()).hexdigest()}
    report['experiment_signature'],report['experiment_sources']=experiment_signature(args.data_root,recipes,args.corrected_folder)
    for source in report['experiment_sources']:
        if source.endswith('.py'):
            snapshot=output/'sources'/Path(source).name;snapshot.parent.mkdir(parents=True,exist_ok=True)
            if not snapshot.exists():snapshot.write_bytes(Path(source).read_bytes())
    fold_selected={};family_predictions={r:{} for r in recipes};family_probabilities={r:{} for r in recipes}
    started=time.perf_counter()
    for fold in chosen_outer:
        val=outer[fold];train=[i for i in all_indices if i not in set(val)]
        inner_local=make_folds([records[i] for i in train],3,20261002+fold*31)
        fold_rows=[]
        for recipe in recipes:
            cachefile=output/'folds'/f'fold{fold}_{recipe}.json'
            if cachefile.is_file():
                row=json.loads(cachefile.read_text(encoding='utf-8'))
                # This output directory is a fixed recipe experiment; never silently consume stale settings.
                if row.get('experiment_signature')!=report['experiment_signature']:raise ValueError('cached experiment signature mismatch; use a new output directory')
                print(f'Reuse fold={fold} recipe={recipe}',flush=True)
            else:
                inner_fp={};inner_vp={};inner_bp={} if "boundary" in recipe else None;t=time.perf_counter()
                for inner_fold,local_val in enumerate(inner_local):
                    inner_val=[train[j] for j in local_val]
                    inner_train=[i for i in train if i not in set(inner_val)]
                    assert not set(inner_train)&set(val) and not set(inner_val)&set(val)
                    fp,vp,state=fit_predict(recipe,records,inner_train,inner_val,20261002+fold*53+inner_fold)
                    if inner_bp is not None:inner_bp.update(state[-1]["boundary_probabilities"])
                    inner_fp.update(fp);inner_vp.update(vp)
                config,inner_result=choose_decoder(records,train,inner_fp,inner_vp,inner_bp)
                print(f'fold={fold} recipe={recipe} inner obj={objective(inner_result):.4f} config={config}',flush=True)
                fp,vp,state=fit_predict(recipe,records,train,val,20261002+fold*53+99)
                predictions,result=evaluate_probabilities(records,val,fp,vp,config,state[-1]["boundary_probabilities"])
                row={'fold':fold,'recipe':recipe,'train_names':[records[i]['name'] for i in train],
                     'validation_names':[records[i]['name'] for i in val],'validation_indices':val,
                     'selection':{'source':'inner 3-fold out-of-video predictions only','config':config,'metrics':inner_result},
                     'inner_frame_probabilities':{str(i):inner_fp[i].tolist() for i in train},
                     'inner_video_probabilities':{str(i):inner_vp[i] for i in train},
                     'inner_boundary_probabilities':{str(i):[v.tolist() for v in inner_bp[i]] for i in train} if inner_bp is not None else None,
                     'boundary_probabilities':{str(i):[v.tolist() for v in state[-1]['boundary_probabilities'][i]] for i in val} if state[-1]['boundary_probabilities'] is not None else None,
                     'validation':result,'predictions':{str(i):[list(v) for v in predictions[i]] for i in val},
                     'frame_probabilities':{str(i):fp[i].tolist() for i in val},'video_probabilities':{str(i):vp[i] for i in val},
                     'elapsed_seconds':time.perf_counter()-t,'dataset_manifest_sha256':report['dataset_manifest_sha256'],
                     'experiment_signature':report['experiment_signature']}
                _json_write(cachefile,row)
            fold_rows.append(row)
            family_predictions[recipe].update({int(i):[tuple(v) for v in pred] for i,pred in row['predictions'].items()})
            family_probabilities[recipe].update({int(i):(np.array(row['frame_probabilities'][i],np.float32),row['video_probabilities'][i]) for i in row['frame_probabilities']})
            print(f"fold={fold} recipe={recipe} OUTER F1.3={row['validation']['iou_0.3']['f1']:.4f} F1.5={row['validation']['iou_0.5']['f1']:.4f} normal={row['validation']['normal']}",flush=True)
        # Family selection is nested too, not selected from this outer fold's score.
        selected=max(fold_rows,key=lambda r:objective(r['selection']['metrics']))
        fold_selected.update({int(i):[tuple(v) for v in pred] for i,pred in selected['predictions'].items()})
        report['folds'].append({'fold':fold,'selected_recipe':selected['recipe'],'selected_by':'inner objective','families':[{k:v for k,v in r.items() if k not in ['frame_probabilities','video_probabilities','inner_frame_probabilities','inner_video_probabilities','boundary_probabilities','inner_boundary_probabilities']} for r in fold_rows]})
        report['elapsed_seconds']=time.perf_counter()-started
        _json_write(output/'progress.json',report)
    if len(fold_selected)==N:
        report['nested_selected_strategy']=_group_metrics(records,fold_selected)
        report['family_oof']={r:_group_metrics(records,p) for r,p in family_predictions.items()}
        report['selected_predictions']=[{'name':r['name'],'segments':[list(v) for v in fold_selected[i]],'labels':r['labels'].astype(int).tolist()} for i,r in enumerate(records)]
        _json_write(output/'report.json',report)
        if args.deployment_fit:
            # Deployment model/threshold selection consumes all development OOF labels.
            # Its in-sample final-fit scores are intentionally not published as validation.
            candidates=[]
            for recipe in recipes:
                fp={i:v[0] for i,v in family_probabilities[recipe].items()};vp={i:v[1] for i,v in family_probabilities[recipe].items()}
                bp=None
                if 'boundary' in recipe:
                    bp={}
                    for fold in chosen_outer:
                        saved=json.loads((output/'folds'/f'fold{fold}_{recipe}.json').read_text(encoding='utf-8'))
                        bp.update({int(i):tuple(np.asarray(v,np.float32) for v in arr) for i,arr in saved['boundary_probabilities'].items()})
                config,result=choose_decoder(records,all_indices,fp,vp,bp)
                candidates.append((objective(result),recipe,config,result))
            _,recipe,config,development_selection=max(candidates,key=lambda v:v[0])
            _,_,(fm,vm,transform,x_values,v_values,extra)=fit_predict(recipe,records,all_indices,all_indices,20261002+8800)
            fm_export=fm if recipe.endswith('_tcn') else export_model(fm);vm_export=export_model(vm)
            # Portable export parity is a hard deployment condition.
            probe=np.concatenate(x_values)
            if not recipe.endswith('_tcn'):
                assert np.max(np.abs(portable_predict(fm_export,probe)-fm.predict_proba(probe)[:,1]))<2e-6
            assert np.max(np.abs(portable_predict(vm_export,v_values)-vm.predict_proba(v_values)[:,1]))<2e-6
            bundle={'schema_version':SCHEMA,'recipe':recipe,'transform':transform,'frame_model':fm_export,'video_model':vm_export,
                    'decoder':config,'boundary_models':extra['boundary_models'],'training_video_sha256':[r['sha256'] for r in records],
                    'training_video_names':[r['name'] for r in records],
                    'training_role':'all inspected development videos; not blind','deployment_selection':development_selection,
                    'evaluation_note':'report.json nested_selected_strategy is held-out by video. Final deployed weights include all development videos; do not use demo examples as generalization evidence.',
                    'feature_profiles':{'jepa':'legacy-event84-relative-v1', **{('corrected' if folder==args.corrected_folder else folder.removesuffix('_features')): json.loads((args.data_root/folder/'manifest.json').read_text(encoding='utf-8')).get('profile') for folder in ['motion_features','rgb_features',args.corrected_folder] if (args.data_root/folder/'manifest.json').is_file()}},
                    'experiment_signature':report['experiment_signature'],
                    'raw_feature_names':{'jepa':records[0]['jepa_names'],'motion':records[0].get('motion_names'),'corrected':records[0].get('corrected_names')},
                    'dataset_manifest_sha256':report['dataset_manifest_sha256'],'trained_at':datetime.now().astimezone().isoformat(timespec='seconds')}
            _json_write(output/'locator_bundle.json',bundle)
            print('DEPLOYMENT recipe',recipe,'config',config,'portable parity passed',flush=True)
    else:_json_write(output/'partial_report.json',report)
    print('Finished',report['elapsed_seconds'],'seconds',flush=True)

if __name__=='__main__':main()