#!/usr/bin/env python3
"""Validated portable inference for new algorithms, separate from research demos."""
from __future__ import annotations
import hashlib, json, re, subprocess, time, uuid
from pathlib import Path
from urllib.request import Request, urlopen, build_opener, ProxyHandler
from urllib.error import URLError
import numpy as np
from optimized_locator import load_bundle, portable_predict, decode, rolling_mean
from optimized_feature_view import feature_view
from optimized_calibration_state import validate_calibration_state, apply_calibration_state
from optimized_motion_features import extract_video_features

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE=ROOT/'models'/'optimized_locator_v1.json'
WORKER_URL='http://127.0.0.1:5004'


def _required_channels_base(bundle):
    if bundle.get('members'):
        return set().union(*(required_channels(m) for m in bundle['members']))
    recipe=bundle['recipe']; result=set()
    if bundle.get('transform',{}).get('feature_view') in ['spatial-jepa-v13','spatial-jepa-mask-repair-v13r']:
        raise ValueError('experimental spatial JEPA cache has no verified live extraction profile; not deployable')
    if bundle.get('transform',{}).get('feature_view')=='compact-feature-blocks-v10':
        from optimized_feature_blocks_v10 import RECIPES
        if recipe not in RECIPES:raise ValueError('unknown feature-block deployment recipe')
        result.update(['motion','local'])
        if 'corrected' in recipe:result.add('corrected')
        if 'rgb' in recipe:result.add('rgb')
        return result
    if bundle.get('transform',{}).get('feature_view')=='compact-fps-context-v4':
        from optimized_compact_features_v4 import RECIPES
        if recipe not in RECIPES:raise ValueError('unknown compact deployment recipe')
        result.update(['motion','local'])
        if 'corrected' in recipe:result.add('corrected')
        if 'rgb' in recipe:result.add('rgb')
        return result
    if recipe.startswith('jepa') or ('hybrid' in recipe and 'corrected' not in recipe):
        raise ValueError('legacy relative JEPA cache has no verified live extraction profile; not deployable')
    if 'motion' in recipe:result.add('motion')
    if 'corrected' in recipe:result.add('corrected')
    if 'local' in recipe:result.add('local')
    if recipe.startswith('rgb_'):result.add('rgb')
    if not result:raise ValueError('unsupported deployment recipe')
    return result


def _validated_proposal_verifier(bundle):
    verifier=bundle.get('proposal_verifier')
    if verifier is None:return None
    if not isinstance(verifier,dict) or set(verifier)!={'schema_version','member','review'} or verifier['schema_version']!='proposal-verifier-v11':
        raise ValueError('unknown proposal verifier schema')
    member=verifier['member']
    if not isinstance(member,dict) or any(k in member for k in ['members','proposal_verifier','decoder']):
        raise ValueError('proposal reviewer must be a leaf member')
    if not all(isinstance(member.get(k),dict) for k in ['transform','frame_model','video_model']):
        raise ValueError('incomplete proposal reviewer leaf schema')
    if member['transform'].get('feature_view')!='compact-fps-context-v4':
        raise ValueError('unverified proposal reviewer feature view')
    from optimized_compact_features_v4 import RECIPES
    if member.get('recipe') not in RECIPES or any(member.get(k) is not None for k in ['calibration','video_gate','boundary_models']):
        raise ValueError('unverified proposal reviewer combination')
    if any(bundle.get(k) is not None for k in ['calibration','video_gate','boundary_models']):
        raise ValueError('proposal verification requires raw uncalibrated base evidence')
    primary=bundle.get('members')
    if primary is not None:
        if not isinstance(primary,list) or len(primary)!=1 or not isinstance(primary[0],dict) or primary[0].get('weight')!=1.:
            raise ValueError('proposal verification requires one primary member')
        if any(primary[0].get(k) is not None for k in ['calibration','video_gate','boundary_models']):
            raise ValueError('unverified primary boundary/calibration combination')
    config=bundle.get('decoder',{})
    if not isinstance(config,dict) or not isinstance(verifier['review'],dict):raise ValueError('invalid proposal policy schema')
    if config.get('kind') not in ['duration-logit-v4','recall-stable-v2'] or config.get('boundary_seconds',0):
        raise ValueError('unverified proposal decoder combination')
    from optimized_proposal_review_v11 import verify_proposals
    verify_proposals([],np.empty(0),np.empty(0),verifier['review'])
    return verifier


def required_channels(bundle):
    result=_required_channels_base(bundle)
    verifier=_validated_proposal_verifier(bundle)
    if verifier is not None:result.update(required_channels(verifier['member']))
    return result


def predict_member(bundle, record):
    if bundle['transform'].get('feature_view')=='compact-feature-blocks-v10':
        from optimized_feature_blocks_v10 import predict_block_member
        fp,vp=predict_block_member(bundle,record)
        return fp,vp,None
    if bundle['transform'].get('feature_view')=='compact-fps-context-v4':
        from optimized_compact_model_v4 import predict_compact_member
        fp,vp=predict_compact_member(bundle,record)
        return fp,vp,None
    if bundle['transform'].get('feature_view')=='shared-label-free-v2':
        from optimized_feature_view_v2 import feature_view_v2
        x,v,names=feature_view_v2(bundle['recipe'],record,bundle['transform'].get('pca'))
    else:x,v,names=feature_view(bundle['recipe'],record,bundle['transform'].get('pca'))
    if names!=bundle['transform']['frame_feature_names']:raise ValueError('trained/inference feature order mismatch')
    model=bundle['frame_model']
    if model.get('kind')=='hist-gradient-boosting-logit-v4':
        from optimized_compact_model_v4 import portable_hgb
        fp=portable_hgb(model,x)
    elif model.get('schema_version')=='optimized-temporal-head-v1':
        from optimized_temporal_head import predict_temporal
        fp=predict_temporal(model,x,record['fps'])
    else:fp=portable_predict(model,x)
    vp=float(portable_predict(bundle['video_model'],v[None])[0])
    boundary=None
    if bundle.get("boundary_models"):
        from optimized_boundary_head import predict_boundary_heads
        boundary=predict_boundary_heads(bundle["boundary_models"],x)
    return fp,vp,boundary


def predict_record(bundle, record):
    calibration=bundle.get('calibration')
    calibration_kind=validate_calibration_state(calibration)
    member_frame=member_video=None
    if calibration_kind=='monotone_stack_logit_v3':
        from optimized_stack_v3 import validate_state
        recipes=[m['recipe'] for m in bundle.get('members',[])]
        if len(recipes)!=len(set(recipes)):raise ValueError('duplicate stack member recipes')
        validate_state(calibration,recipes)
    if bundle.get('members'):
        members=bundle['members'];weights=np.asarray([m['weight'] for m in members],np.float64)
        if not np.isfinite(weights).all() or np.any(weights<0) or not np.isclose(weights.sum(),1):raise ValueError('invalid ensemble weights')
        outputs=[predict_member(m,record) for m in members]
        if calibration_kind=='monotone_stack_logit_v3':
            member_frame={m['recipe']:f for m,(f,v,b) in zip(members,outputs)}
            member_video={m['recipe']:v for m,(f,v,b) in zip(members,outputs)}
        probabilities=sum(w*f for w,(f,v,b) in zip(weights,outputs)).astype(np.float32)
        video_probability=float(sum(w*v for w,(f,v,b) in zip(weights,outputs)))
        boundary_outputs=[(w,b) for w,(f,v,b) in zip(weights,outputs) if b is not None and w>0]
        boundary=None
        if boundary_outputs:
            mass=sum(w for w,b in boundary_outputs)
            boundary=tuple(sum(w*b[j] for w,b in boundary_outputs)/mass for j in [0,1])
    else:probabilities,video_probability,boundary=predict_member(bundle,record)
    raw_video_probability=video_probability
    if bundle.get('video_gate') is not None:
        if bundle['video_gate'].get('numeric_revision') == 'float64-standardization-f32-v6r':
            from optimized_video_gate_v6r import apply_gate
        else:
            from optimized_video_gate_v6 import apply_gate
        video_probability=apply_gate(bundle['video_gate'],probabilities,video_probability,record)
    config=bundle['decoder'];fps=record['fps']
    probabilities,video_probability=apply_calibration_state(probabilities,video_probability,calibration,member_frame=member_frame,member_video=member_video)
    if config.get('kind')=='duration-logit-v4':
        from optimized_duration_decoder_v4 import decode_duration
        intervals=decode_duration(probabilities,fps,video_probability,config)
    elif config.get('kind')=='recall-stable-v2':
        from optimized_recall_decoder import decode_v2
        intervals=decode_v2(probabilities,fps,video_probability,config)
    else:intervals=decode(probabilities,fps,video_probability,config)
    if config.get('boundary_seconds',0):
        if boundary is None:raise ValueError('missing trained boundary evidence')
        from optimized_boundary_head import refine_intervals
        intervals=refine_intervals(intervals,*boundary,fps,config)
    proposal_review_trace=None
    verifier=_validated_proposal_verifier(bundle)
    if verifier is not None:
        from optimized_proposal_review_v11 import verify_proposals
        reviewer,_,_=predict_member(verifier['member'],record)
        intervals,proposal_review_trace=verify_proposals(intervals,probabilities,reviewer,verifier['review'])
    if config.get('kind')=='recall-stable-v2':
        from optimized_recall_decoder import effective_score
        effective=effective_score(probabilities,fps,video_probability,config)
    else:
        effective=probabilities*video_probability**float(config.get('video_strength',0))
        effective=rolling_mean(effective,max(1,int(round(float(config.get('smooth_seconds',0))*fps))))
    result={'frame_probabilities':probabilities,'score':effective,'video_probability':video_probability,'intervals':intervals}
    if bundle.get('video_gate') is not None:
        result.update(raw_video_probability=raw_video_probability,video_gate_kind=bundle['video_gate']['kind'])
    if proposal_review_trace is not None:result['proposal_review_trace']=proposal_review_trace
    return result


def to_wsl(path):
    raw=str(Path(path).resolve());match=re.match(r'^([A-Za-z]):[\\/](.*)$',raw)
    if match:return '/mnt/'+match.group(1).lower()+'/'+match.group(2).replace('\\','/')
    return raw.replace('\\','/')


def from_wsl(path):
    match=re.match(r'^/mnt/([a-z])/(.*)$',str(path))
    return Path(match.group(1).upper()+':\\'+match.group(2).replace('/','\\')) if match else Path(path)


def source_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1048576),b''):h.update(part)
    return h.hexdigest()


def learned_segments(output,fps):
    score=output['score'];segments=[]
    for i,(s,e) in enumerate(output['intervals']):
        segments.append({'id':i+1,'start_frame':int(s),'end_frame':int(e),'frame_count':int(e-s+1),
                         'start_seconds':float(s/fps),'end_seconds':float((e+1)/fps),
                         'duration_seconds':float((e-s+1)/fps),'max_score':float(score[s:e+1].max()),
                         'mean_score':float(score[s:e+1].mean()),'confidence':'uncalibrated_model_evidence',
                         'source':'optimized_learned_locator','note':'end_frame inclusive; seconds end exclusive; learned score is not a calibrated anomaly probability'})
    return segments


def _request_features(video, output, bundle_path, channels, timeout):
    """Use a verified hot worker or a fresh one-shot process; never cached video scores."""
    expected=source_digest(ROOT/'code'/'optimized_model_worker.py')
    local_open=build_opener(ProxyHandler({})).open
    hot=False
    try:
        with local_open(WORKER_URL+'/health',timeout=2) as response:health=json.load(response)
        hot=health.get('worker_code_sha256')==expected and health.get('processing')=='serial'
    except (URLError,OSError,ValueError):pass
    payload={'video':to_wsl(video),'output':to_wsl(output),'channels':sorted(channels),
             'profile_json':to_wsl(bundle_path)}
    if hot:
        request=Request(WORKER_URL+'/infer',data=json.dumps(payload).encode('utf-8'),headers={'Content-Type':'application/json'},method='POST')
        with local_open(request,timeout=timeout) as response:result=json.load(response)
        transport='hot_local_worker'
    else:
        command=['wsl.exe','-d','Ubuntu-24.04','--','/home/zzy/vjepa2-main/vjepa-env/bin/python',
                 to_wsl(ROOT/'code'/'optimized_model_worker.py'),'--video',payload['video'],'--output',payload['output'],
                 '--channels',','.join(sorted(channels)),'--profile-json',payload['profile_json']]
        proc=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout,check=False)
        from demo_detector import _decode_process_bytes
        lines=_decode_process_bytes(proc.stdout).splitlines()
        candidates=[line for line in lines if line.startswith('{') and line.endswith('}')]
        if not candidates:raise RuntimeError('feature worker returned no structured result')
        result=json.loads(candidates[-1]);transport='fresh_one_shot_worker'
        if proc.returncode:raise RuntimeError('feature worker inference failed: '+str(result.get('error',result.get('status'))))
    if result.get('status')!='ok':raise RuntimeError('feature worker rejected inference: '+str(result.get('error',result.get('status'))))
    result['transport']=transport
    return result


def analyze_optimized_video(video_path, output_dir, *, bundle_path=DEFAULT_BUNDLE, timeout=900, progress_callback=None, algorithm='optimized'):
    from demo_detector import _base_report, _render_annotated_video, _video_info, DemoDetectionError
    def progress(message,percent):
        if progress_callback:progress_callback(message,percent)
    started=time.perf_counter();video=Path(video_path).resolve();out=Path(output_dir).resolve()
    out.mkdir(parents=True,exist_ok=True);bundle_path=Path(bundle_path).resolve()
    if not bundle_path.is_file():raise DemoDetectionError('optimized locator bundle is not installed')
    bundle=load_bundle(bundle_path)
    validate_calibration_state(bundle.get('calibration'))
    channels=required_channels(bundle)
    frames,fps,width,height=_video_info(video);record={'fps':fps,'frames':frames};provenance={}
    raw_names=bundle['raw_feature_names']
    progress('提取绝对运动与画质（固定单位，不做视频内拉满）',20)
    if 'motion' in channels:
        motion=extract_video_features(video)
        if len(motion['signals'])!=frames or not motion['validity']['fps_valid'] or not np.isclose(motion['fps'],fps):
            raise DemoDetectionError('motion features are not source-frame/FPS aligned')
        if motion['feature_names']!=raw_names['motion']:raise DemoDetectionError('motion schema differs from fitted model')
        record.update(motion=motion['signals'],motion_names=motion['feature_names'])
        np.savez_compressed(out/'motion_features.npz',signals=motion['signals'],feature_valid=motion['validity']['feature_valid'],fps=np.asarray(fps))
        provenance['motion']={'metadata':motion['metadata'],'feature_names':motion['feature_names']}
    if 'local' in channels:
        from optimized_local_motion import extract_video_features as extract_local
        profile=bundle['feature_profiles'].get('local',{})
        if profile.get('extractor_sha256')!=source_digest(ROOT/'code'/'optimized_local_motion.py'):
            raise DemoDetectionError('local motion extractor differs from trained profile')
        local=extract_local(video,long_edge=profile.get('long_edge',128))
        if not local['validity']['fps_valid'] or len(local['signals'])!=frames or not np.isclose(local['fps'],fps):
            raise DemoDetectionError('local motion frame/FPS mismatch')
        names=['local/'+n for n in local['feature_names']]+['local/'+n+'_support_flag' for n in local['feature_names']]
        if names!=raw_names['local']:raise DemoDetectionError('local motion schema differs from model')
        record.update(local=np.concatenate([local['signals'],local['validity']['feature_valid'].astype(np.float32)],axis=1),local_names=names)
        np.savez_compressed(out/'local_motion_features.npz',signals=local['signals'],feature_valid=local['validity']['feature_valid'],fps=np.asarray(fps))
        provenance['local']={'metadata':local['metadata'],'feature_names':names,'extraction_fresh':True}
    gpu_channels=channels-{'motion','local'};worker_result=None
    if gpu_channels:
        progress('提取新的视频证据（模型复用不复用视频结果）',42)
        # The hot service only permits project-output profiles; this identical
        # snapshot also records precisely which deployed model handled the request.
        profile_copy=out/'active_feature_profile.json'
        profile_copy.write_text(json.dumps({'schema_version':bundle['schema_version'],'feature_profiles':bundle['feature_profiles'],'model_bundle_sha256':source_digest(bundle_path)},ensure_ascii=False,indent=2),encoding='utf-8')
        feature_out=out/('features_'+uuid.uuid4().hex[:10])
        worker_result=_request_features(video,feature_out,profile_copy,gpu_channels,timeout)
        if worker_result['total_frames']!=frames or not np.isclose(worker_result['fps'],fps):raise DemoDetectionError('worker frame/FPS mismatch')
        for channel in gpu_channels:
            info=worker_result['features'][channel];path=from_wsl(info['npz_path']).resolve()
            if not path.is_relative_to(feature_out):raise DemoDetectionError('worker artifact escapes its fresh output directory')
            with np.load(path,allow_pickle=False) as archive:
                key=info.get('feature_key','features' if channel=='rgb' else 'signals')
                values=np.asarray(archive[key],np.float32)
                if channel=='corrected' and archive['names'].tolist()!=raw_names['corrected']:raise DemoDetectionError('fresh corrected feature order differs from model')
            if len(values)!=frames or not np.isfinite(values).all():raise DemoDetectionError('invalid fresh '+channel+' features')
            record[channel]=values
            if channel=='corrected':record['corrected_names']=raw_names['corrected']
        provenance['feature_worker']=worker_result
    progress('学习型定位、视频拒识及多片段精边界',72)
    prediction=predict_record(bundle,record);score=prediction['score'];segments=learned_segments(prediction,fps)
    digest=source_digest(video);seen=digest in bundle.get('training_video_sha256',[])
    true_jepa='corrected' in channels
    method='optimized_true_jepa_motion' if true_jepa else 'optimized_rgb_motion' if 'rgb' in channels else 'optimized_motion'
    label=('Optimized true V/I-JEPA + motion + RGB temporal' if 'rgb' in channels else 'Optimized true V/I-JEPA + learned motion') if true_jepa else 'Frozen R3D + learned motion (not JEPA)' if 'rgb' in channels else ('Learned global + local motion (not JEPA)' if 'local' in channels else 'Learned absolute motion (not JEPA)')
    warnings=['分数是学习型模型证据，未做独立概率校准；空候选不保证视频正常。',
              '模型与参数在已检查的77条开发视频上拟合；交叉验证不是新外部盲测。',
              '这是上传后离线定位，使用了前后文；不承诺在线实时或所有异常都能检测。']
    diagnostic=bool(bundle.get('diagnostic_only',False))
    if diagnostic:warnings.append('此模型是未通过默认部署验收的实验候选；仅供诊断复核，未替换当前演示默认模型。')
    if seen:warnings.append('此视频哈希匹配训练开发集：本次重算验证功能与速度，不代表未见视频泛化。')
    else:warnings.append('此视频未匹配本轮开发集哈希；这本身仍不能算一次规范的新盲测。')
    profile={'name':'optimized-learned-locator-v1','recipe':bundle['recipe'],'decoder':bundle['decoder'],
             'fixed_feature_profile':True,'score_units':'uncalibrated_learned_evidence'}
    report=_base_report(video,method=method,method_label=label,score=score,segments=segments,fps=fps,total_frames=frames,profile=profile,warnings=warnings)
    report.update(algorithm=algorithm,seen_in_development=seen,source_sha256=digest,
                  diagnostic_only=diagnostic,default_deployment_acceptance=bundle.get('default_deployment_acceptance','historical_v1_retained_not_newly_assessed'),
                  comparison_report_sha256=bundle.get('comparison_report_sha256'),
                  model_bundle_sha256=source_digest(bundle_path),model_bundle_path=str(bundle_path),
                  video_evidence_score=prediction['video_probability'],raw_frame_probabilities=prediction['frame_probabilities'].tolist(),
                  feature_provenance=provenance,training_role=bundle.get('training_role'),
                  jepa_usage={'vjepa':'true_masked_predictor' if true_jepa else 'not_used','ijepa':'true_masked_predictor' if true_jepa else 'not_used','proxy_fallback':False})
    progress('生成浏览器可播放的标注视频',88)
    report['annotated_video']=str(out/'annotated.mp4')
    report['video_encoding']=_render_annotated_video(video,out/'annotated.mp4',score,segments,method_label='Optimized learned locator',verdict=report['verdict'],fps=fps,width=width,height=height)
    cfg=bundle.get('feature_profiles',{}).get('corrected',{}).get('config',{})
    report['sampling']={'max_frames':cfg.get('max_frames'),'max_keyframes':cfg.get('max_keyframes'),
                        'random_seed':cfg.get('seed'),'fixed_feature_profile':True,'reproducibility_note':'explicit mask seeds and hidden-counter isolation; verified on this host, not cross-device bitwise promise'}
    report['elapsed_seconds']=round(time.perf_counter()-started,3)
    (out/'demo_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    progress('完成',100)
    return report
