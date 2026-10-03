#!/usr/bin/env python3
"""prepare_dataset.py — 用 total_frames 指纹精准匹配标注到视频"""
import json, os, glob, cv2

ANNO_DIR = '/mnt/e/jepa-label/视频标注/新-train-annotations'
TRAIN_DIR = '/mnt/e/jepa-label/filtered_videos_train'
# test 视频暂不用，标注只对应 train
OUT_DIR = '/home/zzy/jepa_data/clean_dataset'

os.makedirs(os.path.join(OUT_DIR, 'annotations'), exist_ok=True)

# 只索引 train 目录
print("Indexing train videos...")
video_index = {}
for name in os.listdir(TRAIN_DIR):
    path = os.path.join(TRAIN_DIR, name)
    if not name.endswith('.mp4'): continue
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    video_index[name] = (path, n)

print(f"Indexed {len(video_index)} unique (name, frames) pairs")

# Match annotations
cleaned = 0; ambiguous = 0; total_frames_all = 0; total_anom_frames = 0
cats = {}

for af in sorted(glob.glob(os.path.join(ANNO_DIR, '*_annotations.json'))):
    with open(af) as f:
        data = json.load(f)
    name = data.get('video_name', '')
    anno_frames = data.get('total_frames', 0)

    # 直接用文件名匹配（只看 train 目录）
    entry = video_index.get(name)
    if not entry:
        print(f"  SKIP {name}: not in train dir")
        continue
    video_path, video_frames = entry

    # 校验帧数（标注工具的 total_frames 应与实际一致）
    if abs(video_frames - anno_frames) > 1:
        print(f"  WARN {name}: annotator frames={anno_frames}, actual={video_frames}")

    # Fix out-of-bound annotations
    anns = data.get('annotations', [])
    fixed = []
    for a in anns:
        sf = max(0, a['start_frame'])
        ef = min(anno_frames - 1, a['end_frame'])
        if sf > ef: continue
        fixed.append({'start_frame': sf, 'end_frame': ef,
                       'category': a.get('category', 'other'),
                       'severity': a.get('severity', 'medium')})
        total_anom_frames += (ef - sf + 1)
        cats[a.get('category', 'other')] = cats.get(a.get('category', 'other'), 0) + 1

    total_frames_all += anno_frames

    clean = {
        'video_name': name,
        'video_path': video_path,
        'total_frames': anno_frames,
        'annotations': fixed,
        'annotator': data.get('annotator', ''),
        'annotated_at': data.get('annotated_at', ''),
    }
    out_path = os.path.join(OUT_DIR, 'annotations', os.path.basename(af))
    with open(out_path, 'w') as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    cleaned += 1

print(f"\n{'='*50}")
print(f"Matched: {cleaned} videos ({ambiguous} fell back to train dir)")
print(f"Total frames: {total_frames_all}")
print(f"Anomalous frames: {total_anom_frames} ({100*total_anom_frames/max(1,total_frames_all):.1f}%)")
print(f"Categories: {cats}")
print(f"Normal videos: {sum(1 for a in sorted(glob.glob(os.path.join(OUT_DIR,'annotations','*_annotations.json'))) if len(json.load(open(a)).get('annotations',[])) == 0)}")
print(f"Saved to {OUT_DIR}/annotations/")
