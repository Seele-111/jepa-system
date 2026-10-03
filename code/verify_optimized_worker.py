from pathlib import Path
import sys,json,time,numpy as np
sys.path.insert(0,'code')
from optimized_detector import _request_features,from_wsl
root=Path('output/algorithm-opt-2026-10-02').resolve();profile=root/'worker_profile_probe.json'
results=[]
for name,video in [('cold0207',Path(r'C:\Users\admin\Desktop\测试\0207.mp4')),('normal0317',Path(r'C:\Users\admin\Desktop\测试\0317.mp4')),('repeat0207',Path(r'C:\Users\admin\Desktop\测试\0207.mp4')),('short8',root/'short8.mp4')]:
    t=time.perf_counter();r=_request_features(video,root/'worker_smoke'/name,profile,{'rgb','corrected'},900)
    results.append(r)
    print(name,r['status'],'wall',round(time.perf_counter()-t,3),'worker',round(r['elapsed_seconds'],3),'load',r['model_load_status'],flush=True)
arrays=[]
for r in results[:3]:
    with np.load(from_wsl(r['features']['corrected']['npz_path']),allow_pickle=False) as a:arrays.append(a['signals'].copy())
with np.load(root/'corrected_jepa_features_v2'/'v001.npz',allow_pickle=False) as a:cached=a['signals'].copy()
with np.load(root/'corrected_jepa_features_v2'/'v002.npz',allow_pickle=False) as a:cached_normal=a['signals'].copy()
errors={'cold_to_batch':float(np.max(abs(arrays[0]-cached))),'repeat_to_cold':float(np.max(abs(arrays[2]-arrays[0]))),'normal_to_batch':float(np.max(abs(arrays[1]-cached_normal)))}
print('parity',errors,flush=True)
assert max(errors.values())<2e-5
rgb=[]
for r in [results[0],results[2]]:
    with np.load(from_wsl(r['features']['rgb']['npz_path']),allow_pickle=False) as a:rgb.append(a['features'].copy())
with np.load(root/'rgb_features'/'0207.mp4.npz',allow_pickle=False) as a:rgb_cached=a['features'].copy()
errors['rgb_repeat']=float(np.max(abs(rgb[0]-rgb[1])));errors['rgb_to_batch']=float(np.max(abs(rgb[0]-rgb_cached)))
assert max(errors.values())<2e-5
summary={'status':'passed','fresh_features_each_call':all(not r['feature_cache_used'] for r in results),
         'real_gpu':True,'requests':[{'video':r['sourcevideo'],'seconds':r['elapsed_seconds'],'model_load_status':r['model_load_status'],'frames':r['total_frames']} for r in results],
         'max_abs_errors':errors,'short8_frames':results[-1]['total_frames']}
(root/'worker_smoke'/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
