#!/usr/bin/env python3
"""
JEPA 视频异常标注工具 — 手动标记视频异常片段，对接 JEPA 检测流水线
用法: python annotation_tool.py [--port 5003] [--video-dir /path/to/videos]
"""
import os, sys, json, argparse, glob
from datetime import datetime, timedelta
from flask import Flask, render_template_string, request, jsonify, send_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

VIDEO_DIR = ""
ANNO_DIR  = ""
ANNOTATOR = os.environ.get("USER", "annotator")

VIDEO_EXTS = (".mp4", ".webm", ".avi", ".mov", ".mkv")

ANOMALY_CATEGORIES = ["physics", "visual", "other"]


# ── 路由 ────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/videos')
def list_videos():
    videos = []
    for ext in VIDEO_EXTS:
        for p in glob.glob(os.path.join(VIDEO_DIR, f"*{ext}")):
            name = os.path.basename(p)
            if name not in {v["name"] for v in videos}:
                videos.append({"name": name, "path": p})
    # 排序
    videos.sort(key=lambda v: v["name"].lower())

    # 标注进度
    for v in videos:
        anno_path = _anno_path(v["name"])
        v["annotated"] = os.path.exists(anno_path)
        if v["annotated"]:
            try:
                with open(anno_path) as f:
                    a = json.load(f)
                v["segments"] = len(a.get("annotations", []))
            except:
                v["segments"] = 0
        else:
            v["segments"] = 0
    return jsonify(videos)

@app.route('/video/<name>')
def serve_video(name):
    path = _video_path(name)
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path, mimetype="video/mp4")

@app.route('/api/annotation/<name>')
def get_annotation(name):
    path = _anno_path(name)
    if os.path.exists(path):
        with open(path) as f:
            return jsonify(json.load(f))
    # 返回空模板
    video_path = _video_path(name)
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return jsonify({
        "video": video_path,
        "video_name": name,
        "total_frames": total,
        "fps": round(fps, 2) if fps > 0 else 24.0,
        "annotations": [],
        "annotator": ANNOTATOR,
        "annotated_at": None,
    })

@app.route('/api/annotation/<name>/save', methods=['POST'])
def save_annotation(name):
    data = request.get_json()
    data["annotator"] = ANNOTATOR
    data["annotated_at"] = datetime.now().isoformat()
    data["video_name"] = name

    path = _anno_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return jsonify({"ok": True, "path": path})

@app.route('/api/set-video-dir', methods=['POST'])
def set_video_dir():
    global VIDEO_DIR, ANNO_DIR
    data = request.get_json()
    new_dir = data.get('dir', '')
    if not os.path.isdir(new_dir):
        return jsonify({"ok": False, "error": f"目录不存在: {new_dir}"})
    VIDEO_DIR = new_dir
    ANNO_DIR = os.path.join(VIDEO_DIR, "annotations")
    os.makedirs(ANNO_DIR, exist_ok=True)
    return jsonify({"ok": True, "dir": VIDEO_DIR})

@app.route('/api/video-dir')
def get_video_dir():
    return jsonify({"dir": VIDEO_DIR})

@app.route('/api/browse-dir')
def browse_dir():
    """浏览目录结构，返回子目录列表"""
    dirpath = request.args.get('path', '')
    
    # 空路径 = 显示常用根目录
    if not dirpath:
        roots = []
        # Windows 盘符
        for drive in ['C:', 'D:', 'E:', 'F:', 'G:']:
            p = f'/mnt/{drive.lower()[0]}'
            if os.path.isdir(p):
                roots.append({"name": f"💿 {drive}盘", "path": p, "type": "root"})
        # Linux 常用
        home = os.path.expanduser('~')
        if os.path.isdir(home):
            roots.append({"name": "🏠 用户目录", "path": home, "type": "root"})
        roots.append({"name": "📂 根目录", "path": "/", "type": "root"})
        for d in ['/mnt', '/home', '/tmp']:
            if os.path.isdir(d) and d not in [r['path'] for r in roots]:
                roots.append({"name": f"📂 {d}", "path": d, "type": "root"})
        return jsonify({"current": "", "parent": None, "items": roots, "is_root": True})
    
    if not os.path.isdir(dirpath):
        dirpath = os.path.dirname(dirpath) if os.path.exists(os.path.dirname(dirpath)) else os.path.expanduser('~')
    
    items = []
    try:
        for entry in sorted(os.scandir(dirpath), key=lambda e: (not e.is_dir(), e.name.lower())):
            if entry.is_dir() and not entry.name.startswith('.'):
                items.append({
                    "name": entry.name,
                    "path": os.path.abspath(entry.path),
                    "type": "dir"
                })
    except PermissionError:
        pass
    
    parent = os.path.dirname(dirpath) if dirpath != '/' else ''
    return jsonify({
        "current": dirpath,
        "parent": parent,
        "items": items,
        "is_root": False
    })

@app.route('/api/annotation/<name>/backup', methods=['POST'])
def backup_annotation(name):
    data = request.get_json()
    bp = os.path.join(ANNO_DIR, f".{os.path.splitext(name)[0]}_backup.json")
    os.makedirs(os.path.dirname(bp), exist_ok=True)
    with open(bp, 'w') as f:
        json.dump(data, f)
    return jsonify({"ok": True})

@app.route('/api/annotation/<name>/restore')
def restore_annotation(name):
    bp = os.path.join(ANNO_DIR, f".{os.path.splitext(name)[0]}_backup.json")
    if os.path.exists(bp):
        with open(bp) as f:
            anns = json.load(f)
        os.remove(bp)
        return jsonify({"ok": True, "annotations": anns})
    return jsonify({"ok": False, "error": "no backup"})

@app.route('/api/annotation/<name>/has-backup')
def has_backup(name):
    bp = os.path.join(ANNO_DIR, f".{os.path.splitext(name)[0]}_backup.json")
    return jsonify({"has_backup": os.path.exists(bp)})


@app.route('/api/annotation/<name>/jepa')
def get_jepa_comparison(name):
    """加载 JEPA 检测结果作为对照"""
    # 尝试多个可能的路径
    candidates = [
        os.path.join(ANNO_DIR, "..", "detection_summary_v2.json"),
        os.path.join(ANNO_DIR, "..", "detection_summary.json"),
        os.path.join(VIDEO_DIR, "detection_summary_v2.json"),
        os.path.join(VIDEO_DIR, "detection_summary.json"),
    ]
    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            for v in data.get("videos", []):
                if v.get("video_name") == name:
                    return jsonify(v)
        except:
            pass
    return jsonify(None)

@app.route('/api/annotation/<name>/export')
def export_ground_truth(name):
    """导出为 JEPA 训练格式"""
    path = _anno_path(name)
    if not os.path.exists(path):
        return jsonify({"error": "no annotations"}), 404
    with open(path) as f:
        data = json.load(f)

    # 提炼为训练格式
    gt = []
    for ann in data.get("annotations", []):
        gt.append({
            "start_frame": ann["start_frame"],
            "end_frame": ann["end_frame"],
            "category": ann.get("category", ""),
        })
    return jsonify({
        "video": data["video_name"],
        "total_frames": data.get("total_frames", 0),
        "fps": data.get("fps", 0),
        "ground_truth": gt,
        "annotator": data.get("annotator", ""),
    })


# ── 辅助 ────────────────────────────────────────────────

def _video_path(name):
    return os.path.join(VIDEO_DIR, name)

def _anno_path(name):
    stem = os.path.splitext(name)[0]
    return os.path.join(ANNO_DIR, f"{stem}_annotations.json")


# ── HTML 界面 ───────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JEPA 视频异常标注工具</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background:#0d1117; color:#c9d1d9; display:flex; height:100vh; overflow:hidden; }
a { color:#58a6ff; }

/* 左侧视频列表 */
#sidebar { width:280px; min-width:280px; background:#161b22; border-right:1px solid #30363d; display:flex; flex-direction:column; }
#sidebar h2 { padding:16px; font-size:14px; color:#58a6ff; border-bottom:1px solid #30363d; }
#video-list { flex:1; overflow-y:auto; padding:8px; }
.video-item { padding:10px 12px; cursor:pointer; border-radius:6px; margin:2px 0; font-size:12px; display:flex; justify-content:space-between; align-items:center; }
.video-item:hover { background:#1c2128; }
.video-item.active { background:#1f6feb22; border-left:3px solid #58a6ff; padding-left:9px; }
.video-item .badge { font-size:10px; padding:2px 6px; border-radius:8px; }
.video-item .done { background:#23863622; color:#3fb950; }
.video-item .pending { color:#8b949e; }
#sidebar .info { padding:12px 16px; font-size:11px; color:#8b949e; border-top:1px solid #30363d; }

/* 主区域 */
#main { flex:1; display:flex; flex-direction:column; min-width:0; }

/* 顶部工具栏 */
#toolbar { padding:12px 16px; border-bottom:1px solid #30363d; display:flex; gap:8px; align-items:center; flex-wrap:wrap; font-size:13px; }
#toolbar button { padding:6px 14px; border:1px solid #30363d; background:#21262d; color:#c9d1d9; border-radius:6px; cursor:pointer; font-size:12px; }
#toolbar button:hover { background:#30363d; }
#toolbar button.primary { background:#238636; border-color:#238636; color:#fff; }
#toolbar button.danger { background:#da363322; border-color:#da3633; color:#da3633; }
#toolbar select { padding:6px 10px; background:#21262d; color:#c9d1d9; border:1px solid #30363d; border-radius:6px; font-size:12px; }
#toolbar .sep { width:1px; height:20px; background:#30363d; margin:0 4px; }
#toolbar .status { margin-left:auto; font-size:11px; color:#8b949e; }

/* 视频播放器 */
#player-area { flex:1; display:flex; flex-direction:column; background:#000; position:relative; min-height:300px; align-items:center; justify-content:center; }
#player-area .placeholder { color:#484f58; font-size:14px; }
video { max-width:100%; max-height:100%; }
canvas { position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; }

/* 帧控制 */
#frame-bar { display:flex; align-items:center; padding:8px 16px; background:#161b22; border-top:1px solid #30363d; gap:8px; }
#frame-bar button { padding:4px 10px; background:#21262d; color:#c9d1d9; border:1px solid #30363d; border-radius:4px; cursor:pointer; font-size:11px; }
#frame-bar button:hover { background:#30363d; }
#frame-bar .frame-display { font-size:13px; font-family:monospace; color:#58a6ff; min-width:120px; text-align:center; }
#frame-bar input[type=range] { flex:1; accent-color:#58a6ff; }
#frame-bar .jump { width:60px; padding:4px; background:#0d1117; color:#c9d1d9; border:1px solid #30363d; border-radius:4px; text-align:center; font-size:11px; }

/* 标注面板 */
#anno-panel { background:#161b22; border-top:1px solid #30363d; padding:12px 16px; display:flex; gap:12px; flex-wrap:wrap; align-items:center; font-size:12px; }
#anno-panel button { padding:6px 14px; border-radius:6px; cursor:pointer; font-size:12px; border:1px solid #30363d; }
#anno-panel .set-start { background:#1f6feb; border-color:#1f6feb; color:#fff; }
#anno-panel .set-end { background:#da3633; border-color:#da3633; color:#fff; }
#anno-panel .add-seg { background:#238636; border-color:#238636; color:#fff; }
#anno-panel select { padding:6px 10px; background:#21262d; color:#c9d1d9; border:1px solid #30363d; border-radius:6px; }
.seg-preview { font-family:monospace; color:#d2a8ff; font-size:13px; }

/* 标注列表 */
#segments-list { border-top:1px solid #30363d; max-height:200px; overflow-y:auto; padding:8px 16px; background:#0d1117; }
.seg-row { display:flex; align-items:center; gap:8px; padding:6px 8px; font-size:12px; border-bottom:1px solid #21262d; }
.seg-row:hover { background:#161b22; }
.seg-row .seg-info { flex:1; }
.seg-row .seg-cat { font-size:10px; padding:1px 6px; border-radius:6px; background:#1f6feb22; color:#58a6ff; }
.seg-row .seg-cat.visual { background:#da363322; color:#da3633; }
.seg-row button { padding:2px 8px; font-size:10px; background:transparent; border:1px solid #30363d; color:#8b949e; border-radius:4px; cursor:pointer; }
.seg-row button:hover { color:#da3633; border-color:#da3633; }

/* 键盘快捷键提示 */
#shortcuts { position:fixed; bottom:70px; right:16px; background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px; font-size:11px; color:#8b949e; display:none; z-index:100; }
#shortcuts kbd { background:#21262d; padding:1px 5px; border-radius:3px; border:1px solid #30363d; font-family:monospace; color:#c9d1d9; }
#shortcuts .row { margin:3px 0; }

/* 目录浏览器弹窗 */
#browser-modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.7); z-index:999; justify-content:center; align-items:center; }
#browser-modal.show { display:flex; }
#browser-box { background:#161b22; border:1px solid #30363d; border-radius:8px; width:500px; max-height:500px; display:flex; flex-direction:column; }
#browser-box .b-header { padding:12px; border-bottom:1px solid #30363d; display:flex; gap:8px; align-items:center; }
#browser-box .b-header .b-path { flex:1; font-size:11px; color:#8b949e; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
#browser-box .b-list { flex:1; overflow-y:auto; padding:4px; }
#browser-box .b-item { padding:8px 12px; cursor:pointer; border-radius:4px; font-size:12px; display:flex; align-items:center; gap:6px; }
#browser-box .b-item:hover { background:#1c2128; }
#browser-box .b-item .icon { font-size:14px; }
#browser-box .b-footer { padding:10px; border-top:1px solid #30363d; display:flex; gap:8px; justify-content:flex-end; }
#browser-box .b-footer button { padding:5px 14px; border-radius:4px; border:1px solid #30363d; cursor:pointer; font-size:12px; background:#21262d; color:#c9d1d9; }
#browser-box .b-footer button.ok { background:#238636; border-color:#238636; color:#fff; }
</style>
</head>
<body onload="fetch('/api/video-dir').then(r=>r.json()).then(d=>document.getElementById('dir-input').value=d.dir)">

<div id="sidebar">
  <h2>📁 视频列表</h2>
  <div style="padding:8px;border-bottom:1px solid #30363d;">
    <div style="display:flex;gap:4px;">
      <input id="dir-input" style="flex:1;padding:6px 8px;background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;font-size:11px;" placeholder="视频文件夹路径...">
      <button onclick="openBrowser()" style="padding:6px 8px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;cursor:pointer;font-size:11px;">📁</button>
    </div>
    <button onclick="switchDir()" style="margin-top:4px;width:100%;padding:5px;background:#21262d;color:#c9d1d9;border:1px solid #30363d;border-radius:4px;cursor:pointer;font-size:11px;">📂 切换文件夹</button>
  </div>
  <div id="video-list"></div>
  <div class="info" id="sidebar-info">加载中...</div>
</div>

<div id="main">
  <div id="toolbar">
    <span id="current-video" style="font-weight:600;">请选择视频</span>
    <span class="sep"></span>
    <button onclick="saveAnnotations()" class="primary">💾 保存</button>
    <button onclick="togglePlay()" id="btn-play">▶ 播放</button>
    <button id="btn-normal" onclick="markNormal()">✓ 此视频正常</button>
    <button onclick="loadJepaComparison()">🔬 JEPA 对照</button>
    <button id="btn-jepa-toggle" onclick="toggleJepaOverlay()" style="display:none">📍 显示JEPA</button>
    <button onclick="exportDataset()" style="background:#6e40c9;border-color:#6e40c9;color:#fff">📦 导出训练数据</button>
    <span class="sep"></span>
    <select id="category">
      <option value="physics">physics（物理异常）</option>
      <option value="visual">visual（视觉伪影）</option>
      <option value="other">other（其他）</option>
    </select>
    <span class="sep"></span>
    <select id="severity">
      <option value="high">high</option>
      <option value="medium" selected>medium</option>
      <option value="low">low</option>
    </select>
    <span class="status" id="save-status"></span>
  </div>

  <div id="player-area">
    <video id="video" crossorigin="anonymous" style="display:none"></video>
    <canvas id="overlay-canvas" style="display:none"></canvas>
    <div class="placeholder" id="placeholder">← 选择视频开始标注</div>
  </div>

  <div id="frame-bar">
    <button onclick="stepFrame(-5)">⏪ -5</button>
    <button onclick="stepFrame(-1)">◀ -1</button>
    <span class="frame-display" id="frame-display">0 / 0</span>
    <button onclick="stepFrame(1)">+1 ▶</button>
    <button onclick="stepFrame(5)">+5 ⏩</button>
    <input type="range" id="frame-slider" min="0" max="100" value="0" oninput="seekToFrame(this.value)">
    <input class="jump" id="jump-input" placeholder="跳转" onkeydown="if(event.key==='Enter') jumpToFrame()">
  </div>

  <div id="anno-panel">
    <button class="set-start" onclick="setStart()" id="btn-start">⏺ 标记起点</button>
    <button class="set-end" onclick="setEnd()" id="btn-end">⏹ 标记终点</button>
    <span class="seg-preview" id="seg-preview">未选择</span>
    <button class="add-seg" onclick="addSegment()" id="btn-add" disabled>✅ 添加异常段</button>
  </div>

  <div id="segments-list"></div>
</div>

<div id="browser-modal">
  <div id="browser-box">
    <div class="b-header">
      <span>📁 选择文件夹</span>
      <input id="browser-jump" style="flex:1;padding:4px 8px;background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:3px;font-size:11px;" placeholder="输入路径回车直达..." onkeydown="if(event.key==='Enter')navigateBrowser(this.value)">
    </div>
    <div class="b-header" style="border-top:none;padding-top:0;">
      <span class="b-path" id="browser-path">选择文件夹</span>
    </div>
    <div class="b-list" id="browser-list"></div>
    <div class="b-footer">
      <button onclick="closeBrowser()">取消</button>
      <button class="ok" onclick="selectBrowserDir()">✓ 选择此文件夹</button>
    </div>
  </div>
</div>

<div id="shortcuts">
  <div class="row"><kbd>Space</kbd> 播放/暂停</div>
  <div class="row"><kbd>← →</kbd> 前后 1 帧</div>
  <div class="row"><kbd>Shift+← →</kbd> 前后 5 帧</div>
  <div class="row"><kbd>S</kbd> 标记起点</div>
  <div class="row"><kbd>E</kbd> 标记终点</div>
  <div class="row"><kbd>Enter</kbd> 添加异常段</div>
  <div class="row"><kbd>Ctrl+S</kbd> 保存</div>
</div>

<script>
let currentVideo = null;
let annotations = null;
let startFrame = null;
let endFrame = null;
let currentFrame = 0;
let jepaData = null;
let showJepa = false;

const video = document.getElementById('video');
const canvas = document.getElementById('overlay-canvas');
const ctx = canvas.getContext('2d');

let browserCurrent = '/';

function openBrowser() {
  document.getElementById('browser-modal').classList.add('show');
  // 不传路径 = 显示常用根目录
  navigateBrowser('');
}

function closeBrowser() {
  document.getElementById('browser-modal').classList.remove('show');
}

function navigateBrowser(dirpath) {
  browserCurrent = dirpath;
  let url = '/api/browse-dir';
  if (dirpath) url += '?path='+encodeURIComponent(dirpath);
  fetch(url).then(r=>r.json()).then(data => {
    document.getElementById('browser-path').textContent = data.current || '选择文件夹';
    const list = document.getElementById('browser-list');
    let html = '';
    if (data.parent) {
      html += `<div class="b-item" onclick="navigateBrowser('${data.parent}')"><span class="icon">📂</span> .. 上级目录</div>`;
    }
    data.items.forEach(item => {
      const icon = item.type === 'root' ? '' : '📁 ';
      html += `<div class="b-item" onclick="navigateBrowser('${item.path}')">${icon}${item.name}</div>`;
    });
    if (!data.parent && data.items.length === 0 && !data.is_root) {
      html = '<div style="padding:20px;text-align:center;color:#484f58;">此目录下无子目录</div>';
    }
    list.innerHTML = html;
  });
}

function selectBrowserDir() {
  if (!browserCurrent) { alert('请先进入一个具体目录'); return; }
  document.getElementById('dir-input').value = browserCurrent;
  closeBrowser();
  switchDir();
}

function switchDir() {
  const p = document.getElementById('dir-input').value.trim();
  if (!p) return;
  setStatus('切换中...', '#d29922');
  fetch('/api/set-video-dir', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({dir:p})})
    .then(r=>r.json()).then(d=>{
      if (d.ok) {
        currentVideo = null; annotations = null;
        video.src = ''; video.style.display = 'none';
        document.getElementById('placeholder').style.display = 'block';
        document.getElementById('current-video').textContent = '请选择视频';
        refreshVideoList();
        setStatus('已切换到: '+d.dir, '#3fb950');
      } else {
        alert('切换失败: '+d.error);
        setStatus('切换失败', '#da3633');
      }
    }).catch(e => { alert('请求失败: '+e); setStatus('错误', '#da3633'); });
}

function refreshVideoList() {
  fetch('/api/videos?_='+Date.now()).then(r=>r.json()).then(videos => {
    document.getElementById('sidebar-info').textContent = `${videos.length} 个视频`;
    const list = document.getElementById('video-list');
    list.innerHTML = '';
    videos.forEach(v => {
      const div = document.createElement('div');
      div.className = 'video-item';
      div.innerHTML = `<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:180px">${v.name}</span>
        <span class="badge ${v.annotated ? 'done' : 'pending'}">${v.annotated ? v.segments+'段' : '未标注'}</span>`;
      div.onclick = () => loadVideo(v.name);
      list.appendChild(div);
    });
  });
}

// ── 初始化 ──
fetch('/api/videos').then(r=>r.json()).then(videos => {
  document.getElementById('sidebar-info').textContent = `${videos.length} 个视频`;
  const list = document.getElementById('video-list');
  videos.forEach(v => {
    const div = document.createElement('div');
    div.className = 'video-item';
    div.innerHTML = `
      <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:180px">${v.name}</span>
      <span class="badge ${v.annotated ? 'done' : 'pending'}">${v.annotated ? v.segments+'段' : '未标注'}</span>
    `;
    div.onclick = () => loadVideo(v.name);
    list.appendChild(div);
  });
});

// ── 视频控制 ──
function loadVideo(name) {
  currentVideo = name;
  document.getElementById('current-video').textContent = name;
  document.getElementById('placeholder').style.display = 'none';
  video.style.display = 'block';
  video.src = `/video/${encodeURIComponent(name)}`;
  video.load();

  // 高亮当前
  document.querySelectorAll('.video-item').forEach(el => el.classList.remove('active'));
  Array.from(document.querySelectorAll('.video-item')).find(el => el.textContent.includes(name))?.classList.add('active');

  // 加载标注
  fetch(`/api/annotation/${encodeURIComponent(name)}`).then(r=>r.json()).then(data => {
    annotations = data;
    document.getElementById('frame-slider').max = data.total_frames - 1;
    updateSegmentsList();
    updateFrameDisplay();
    setStatus('已加载');
    fetch('/api/annotation/'+encodeURIComponent(name)+'/has-backup').then(r=>r.json()).then(bk => {
      const btn = document.getElementById('btn-normal');
      if (bk.has_backup) {
        annotations._has_backup = true;
        btn.textContent = '↩ 撤销正常';
        btn.style.color = '#3fb950';
      } else {
        annotations._has_backup = false;
        btn.textContent = '✓ 此视频正常';
        btn.style.color = '';
      }
    });
  });

  jepaData = null;
  showJepa = false;
  document.getElementById('btn-jepa-toggle').style.display = 'none';
  startFrame = null; endFrame = null;
  updateSegPreview();
}

video.addEventListener('loadedmetadata', () => {
  document.getElementById('frame-slider').max = Math.floor(video.duration * (annotations?.fps || 24)) - 1;
});

video.addEventListener('timeupdate', () => {
  if (!annotations) return;
  const fps = annotations.fps || 24;
  currentFrame = Math.round(video.currentTime * fps);
  document.getElementById('frame-slider').value = currentFrame;
  updateFrameDisplay();
  drawJepaOverlay();
});

function togglePlay() {
  const btn = document.getElementById('btn-play');
  if (video.paused) {
    video.play(); btn.textContent = '⏸ 暂停';
  } else {
    video.pause(); btn.textContent = '▶ 播放';
  }
}
video.addEventListener('play', ()=>document.getElementById('btn-play').textContent='⏸ 暂停');
video.addEventListener('pause', ()=>document.getElementById('btn-play').textContent='▶ 播放');

function stepFrame(delta) {
  if (!annotations) return;
  const fps = annotations.fps || 24;
  currentFrame = Math.max(0, Math.min(annotations.total_frames - 1, currentFrame + delta));
  video.currentTime = currentFrame / fps;
}

function seekToFrame(f) {
  if (!annotations) return;
  currentFrame = parseInt(f);
  video.currentTime = currentFrame / (annotations.fps || 24);
}

function jumpToFrame() {
  const v = parseInt(document.getElementById('jump-input').value);
  if (!isNaN(v) && annotations) {
    currentFrame = Math.max(0, Math.min(annotations.total_frames - 1, v));
    video.currentTime = currentFrame / (annotations.fps || 24);
  }
}

function updateFrameDisplay() {
  if (!annotations) return;
  const fps = annotations.fps || 24;
  const secs = currentFrame / fps;
  const ts = new Date(secs * 1000).toISOString().substr(11, 8);
  document.getElementById('frame-display').textContent = `帧 ${currentFrame} / ${annotations.total_frames - 1} | ${ts}`;
}

// ── 标注操作 ──
function setStart() {
  startFrame = currentFrame;
  updateSegPreview();
  document.getElementById('btn-add').disabled = (startFrame === null || endFrame === null);
}

function setEnd() {
  endFrame = currentFrame;
  updateSegPreview();
  document.getElementById('btn-add').disabled = (startFrame === null || endFrame === null);
}

function updateSegPreview() {
  const el = document.getElementById('seg-preview');
  if (startFrame !== null && endFrame !== null) {
    const fps = annotations?.fps || 24;
    el.textContent = `起点:${startFrame} → 终点:${endFrame} (${endFrame-startFrame+1}帧 / ${((endFrame-startFrame+1)/fps).toFixed(2)}s)`;
    el.style.color = '#3fb950';
  } else if (startFrame !== null) {
    el.textContent = `起点:${startFrame} → 终点:?`;
    el.style.color = '#d2a8ff';
  } else {
    el.textContent = '未选择';
    el.style.color = '#8b949e';
  }
}

function addSegment() {
  if (startFrame === null || endFrame === null) return;
  if (endFrame < startFrame) [startFrame, endFrame] = [endFrame, startFrame];

  const fps = annotations.fps || 24;
  const seg = {
    start_frame: startFrame,
    end_frame: endFrame,
    timestamp_start: new Date(startFrame / fps * 1000).toISOString().substr(11, 12),
    timestamp_end: new Date(endFrame / fps * 1000).toISOString().substr(11, 12),
    category: document.getElementById('category').value,
    severity: document.getElementById('severity').value,
    notes: ""
  };

  annotations.annotations.push(seg);
  startFrame = null; endFrame = null;
  document.getElementById('btn-add').disabled = true;
  updateSegmentsList();
  updateSegPreview();
  saveAnnotations(false);
  setStatus('已添加', '#3fb950');
}

function deleteSegment(idx) {
  annotations.annotations.splice(idx, 1);
  updateSegmentsList();
  saveAnnotations(false);
  setStatus('已删除', '#3fb950');
}

function updateSegmentsList() {
  const list = document.getElementById('segments-list');
  const segs = annotations?.annotations || [];
  list.innerHTML = segs.map((s, i) => `
    <div class="seg-row">
      <span class="seg-info">
        帧 ${s.start_frame}-${s.end_frame} | ${s.timestamp_start}→${s.timestamp_end}
        <span class="seg-cat ${s.category==='visual'?'visual':''}">${s.category}</span>
        [${s.severity}]
      </span>
      <button onclick="deleteSegment(${i})" title="删除">✕</button>
    </div>
  `).join('') || '<div style="padding:12px;color:#484f58;font-size:12px;">暂无标注段</div>';
}


// ── 保存 ──
function saveAnnotations(reloadAfter) {
  if (!currentVideo || !annotations) return;
  fetch(`/api/annotation/${encodeURIComponent(currentVideo)}/save`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(annotations)
  }).then(r=>r.json()).then(() => {
    setStatus('已保存 ✓', '#3fb950');
    refreshVideoList();
    if (reloadAfter) setTimeout(() => location.reload(), 500);
  });
}

function markNormal() {
  if (!currentVideo || !annotations) return;
  const btn = document.getElementById('btn-normal');

  // 撤销模式
  if (annotations.annotations.length === 0 && annotations._has_backup) {
    fetch('/api/annotation/'+encodeURIComponent(currentVideo)+'/restore')
      .then(r=>r.json()).then(data => {
        if (data.ok) {
          annotations.annotations = data.annotations;
          annotations._has_backup = false;
          updateSegmentsList();
          saveAnnotations(false);
          btn.textContent = '✓ 此视频正常';
          setStatus('已恢复标注', '#58a6ff');
        }
      });
    return;
  }

  // 标记正常模式
  if (annotations.annotations.length > 0) {
    if (!confirm('视频已有 '+annotations.annotations.length+' 个异常段，确认清空并标记为正常？')) return;
  }

  setStatus('保存中...', '#d29922');
  // 先存备份，完成后才清空
  fetch('/api/annotation/'+encodeURIComponent(currentVideo)+'/backup', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(annotations.annotations)
  }).then(r=>r.json()).then(() => {
    annotations.annotations = [];
    annotations._has_backup = true;
    updateSegmentsList();
    saveAnnotations(false);
    btn.textContent = '↩ 撤销正常';
    btn.style.color = '#3fb950';
    setStatus('已标记为正常（可撤销）', '#3fb950');
  });
}

function exportDataset() {
  fetch('/api/export/dataset').then(r=>r.blob()).then(blob=>{
    const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
    a.download='jepa_training_data.zip'; a.click();
    setStatus('训练数据已下载', '#3fb950');
  }).catch(()=>setStatus('导出失败', '#da3633'));
}

function setStatus(msg, color) {
  const el = document.getElementById('save-status');
  el.textContent = msg;
  el.style.color = color || '#8b949e';
  if (color === '#3fb950') setTimeout(() => setStatus(''), 3000);
}

// ── JEPA 对照 ──
function loadJepaComparison() {
  if (!currentVideo) return;
  fetch(`/api/annotation/${encodeURIComponent(currentVideo)}/jepa`).then(r=>r.json()).then(data => {
    if (data && data.segments) {
      jepaData = data;
      document.getElementById('btn-jepa-toggle').style.display = 'inline';
      document.getElementById('btn-jepa-toggle').textContent = '📍 显示JEPA检出';
      showJepa = true;
      drawJepaOverlay();
      setStatus(`JEPA检出: ${data.segments.length}段`, '#58a6ff');
    } else {
      setStatus('无JEPA检测结果', '#8b949e');
    }
  });
}

function toggleJepaOverlay() {
  showJepa = !showJepa;
  document.getElementById('btn-jepa-toggle').textContent = showJepa ? '📍 隐藏JEPA' : '📍 显示JEPA';
  if (!showJepa) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.style.display = 'none';
  } else {
    drawJepaOverlay();
  }
}

function drawJepaOverlay() {
  if (!showJepa || !jepaData || !annotations) return;
  canvas.style.display = 'block';
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 显示JEPA异常段和人工标注段的对比
  const fps = annotations.fps || 24;

  // JEPA 段 — 蓝色条
  (jepaData.segments || []).forEach(seg => {
    const y = canvas.height - 30;
    const w = canvas.width;
    const x1 = seg.start_frame / annotations.total_frames * w;
    const x2 = seg.end_frame / annotations.total_frames * w;
    ctx.fillStyle = 'rgba(88, 166, 255, 0.3)';
    ctx.fillRect(x1, y, x2 - x1, 14);
    ctx.fillStyle = '#58a6ff';
    ctx.font = '10px monospace';
    ctx.fillText(`JEPA:${seg.start_frame}-${seg.end_frame}`, x1 + 2, y + 11);
  });

  // 人工标注段 — 绿色条
  (annotations.annotations || []).forEach(seg => {
    const y = canvas.height - 14;
    const w = canvas.width;
    const x1 = seg.start_frame / annotations.total_frames * w;
    const x2 = seg.end_frame / annotations.total_frames * w;
    ctx.fillStyle = 'rgba(63, 185, 80, 0.3)';
    ctx.fillRect(x1, y, x2 - x1, 14);
    ctx.fillStyle = '#3fb950';
    ctx.font = '10px monospace';
    ctx.fillText(`标注:${seg.start_frame}-${seg.end_frame}`, x1 + 2, y + 11);
  });
}

// ── 键盘快捷键 ──
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.ctrlKey && e.key === 's') { e.preventDefault(); saveAnnotations(true); return; }

  switch(e.key) {
    case ' ': e.preventDefault(); video.paused ? video.play() : video.pause(); break;
    case 'ArrowLeft': e.preventDefault(); stepFrame(e.shiftKey ? -5 : -1); break;
    case 'ArrowRight': e.preventDefault(); stepFrame(e.shiftKey ? 5 : 1); break;
    case 's': case 'S': setStart(); break;
    case 'e': case 'E': setEnd(); break;
    case 'Enter': addSegment(); break;
  }
});
</script>
</body>
</html>"""



@app.route('/api/export/dataset')
def export_dataset():
    """导出训练数据集：per-frame labels + segment annotations + train/val split"""
    import zipfile, io, cv2 as _cv2

    buf = io.BytesIO()
    zf = zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED)

    # 收集所有标注
    all_videos = []
    per_frame = {}
    anno_files = glob.glob(os.path.join(ANNO_DIR, "*_annotations.json"))

    for af in sorted(anno_files):
        with open(af) as f:
            data = json.load(f)
        name = data.get("video_name", "")
        total = data.get("total_frames", 0)
        anns = data.get("annotations", [])

        # Per-frame binary labels
        labels = [0] * total
        for a in anns:
            for f in range(a["start_frame"], a["end_frame"] + 1):
                if 0 <= f < total:
                    labels[f] = 1

        per_frame[name] = labels
        all_videos.append({
            "video_name": name,
            "total_frames": total,
            "fps": data.get("fps", 0),
            "annotations": anns,
            "has_anomaly": len(anns) > 0,
        })

    # Train/Val split (80/20)
    import random
    random.seed(42)
    shuffled = all_videos[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * 0.8)
    train_names = [v["video_name"] for v in shuffled[:split]]
    val_names = [v["video_name"] for v in shuffled[split:]]

    zf.writestr("per_frame_labels.json", json.dumps(per_frame, indent=2))
    zf.writestr("annotations_summary.json", json.dumps(all_videos, indent=2, ensure_ascii=False))
    zf.writestr("train_val_split.json", json.dumps({
        "train": train_names, "val": val_names,
        "train_count": len(train_names), "val_count": len(val_names),
    }, indent=2))

    # JEPA 评估
    jepa_eval = _evaluate_jepa(all_videos)
    zf.writestr("jepa_evaluation.json", json.dumps(jepa_eval, indent=2, ensure_ascii=False))

    # README
    zf.writestr("README.txt", """JEPA 训练数据集
================

文件说明:
  per_frame_labels.json     — 逐帧标签 (0=正常, 1=异常), key=视频名
  annotations_summary.json  — 段级标注详情
  train_val_split.json      — 训练/验证集划分 (80/20)
  jepa_evaluation.json      — JEPA检测 vs 人工标注 对比指标

使用:
  import json
  with open('per_frame_labels.json') as f:
      labels = json.load(f)  # {video_name: [0,0,1,1,1,0,...]}
  with open('train_val_split.json') as f:
      split = json.load(f)   # {'train': [...], 'val': [...]}
""".encode("utf-8"))

    zf.close()
    buf.seek(0)

    from flask import Response
    return Response(buf, mimetype="application/zip",
                    headers={"Content-Disposition": "attachment; filename=jepa_training_data.zip"})


def _evaluate_jepa(videos_annotated):
    """对比 JEPA 检测结果与人工标注"""
    import random
    # 尝试加载 JEPA 汇总
    candidates = [
        os.path.join(ANNO_DIR, "..", "detection_summary_v2.json"),
        os.path.join(ANNO_DIR, "..", "detection_summary.json"),
        os.path.join(VIDEO_DIR, "detection_summary_v2.json"),
        os.path.join(VIDEO_DIR, "detection_summary.json"),
    ]
    jepa_data = None
    for cp in candidates:
        if os.path.exists(cp):
            with open(cp) as f:
                jepa_data = json.load(f)
            break

    if not jepa_data:
        return {"error": "未找到 JEPA 检测结果文件"}

    jepa_map = {}
    for v in jepa_data.get("videos", []):
        jepa_map[v["video_name"]] = [(s["start_frame"], s["end_frame"]) for s in v.get("segments", [])]

    total = 0
    tp = fp = fn = 0
    frame_tp = frame_fp = frame_fn = 0
    per_video = []

    for va in videos_annotated:
        name = va["video_name"]
        gt_segs = [(a["start_frame"], a["end_frame"]) for a in va.get("annotations", [])]
        jp_segs = jepa_map.get(name, [])
        total_frames = va.get("total_frames", 0)

        # Frame-level
        gt_frames = set()
        for s, e in gt_segs:
            gt_frames.update(range(s, e + 1))
        jp_frames = set()
        for s, e in jp_segs:
            jp_frames.update(range(s, e + 1))

        f_tp = len(gt_frames & jp_frames)
        f_fp = len(jp_frames - gt_frames)
        f_fn = len(gt_frames - jp_frames)
        frame_tp += f_tp
        frame_fp += f_fp
        frame_fn += f_fn

        # Segment-level IoU matching
        matched_gt = set()
        matched_jp = set()
        for gi, (gs, ge) in enumerate(gt_segs):
            for ji, (js, je) in enumerate(jp_segs):
                inter = max(0, min(ge, je) - max(gs, js))
                union = max(ge, je) - min(gs, js)
                if union > 0 and inter / union > 0.3:
                    matched_gt.add(gi)
                    matched_jp.add(ji)

        tp += len(matched_gt)
        fp += len(jp_segs) - len(matched_jp)
        fn += len(gt_segs) - len(matched_gt)
        total += 1

        f_prec = f_tp / (f_tp + f_fp) if (f_tp + f_fp) > 0 else 0
        f_rec = f_tp / (f_tp + f_fn) if (f_tp + f_fn) > 0 else 0
        per_video.append({
            "video": name,
            "gt_segments": len(gt_segs),
            "jepa_segments": len(jp_segs),
            "frame_precision": round(f_prec, 3),
            "frame_recall": round(f_rec, 3),
            "frame_f1": round(2*f_prec*f_rec/(f_prec+f_rec), 3) if (f_prec+f_rec)>0 else 0,
        })

    f_prec = frame_tp / (frame_tp + frame_fp) if (frame_tp + frame_fp) > 0 else 0
    f_rec = frame_tp / (frame_tp + frame_fn) if (frame_tp + frame_fn) > 0 else 0
    s_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    s_rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "total_videos": total,
        "annotated_videos": sum(1 for v in videos_annotated if v.get("annotations")),
        "frame_level": {
            "precision": round(f_prec, 3),
            "recall": round(f_rec, 3),
            "f1": round(2*f_prec*f_rec/(f_prec+f_rec), 3) if (f_prec+f_rec) > 0 else 0,
            "true_positive_frames": frame_tp,
            "false_positive_frames": frame_fp,
            "false_negative_frames": frame_fn,
        },
        "segment_level": {
            "precision": round(s_prec, 3),
            "recall": round(s_rec, 3),
            "f1": round(2*s_prec*s_rec/(s_prec+s_rec), 3) if (s_prec+s_rec) > 0 else 0,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        },
        "per_video": per_video,
        "jepa_source": candidates[0] if jepa_data else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JEPA 视频异常标注工具")
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument("--video-dir", default=None, help="视频文件夹")
    parser.add_argument("--anno-dir", default=None, help="标注输出目录")
    parser.add_argument("--annotator", default=ANNOTATOR, help="标注者名称")
    args = parser.parse_args()

    VIDEO_DIR = os.path.abspath(args.video_dir or os.getcwd())
    ANNO_DIR = os.path.abspath(args.anno_dir or os.path.join(VIDEO_DIR, "annotations"))
    ANNOTATOR = args.annotator

    os.makedirs(ANNO_DIR, exist_ok=True)

    print(f"""
╔══════════════════════════════════════════════════╗
║       JEPA 视频异常标注工具                       ║
╠══════════════════════════════════════════════════╣
║  视频目录: {VIDEO_DIR}
║  标注目录: {ANNO_DIR}
║  标注者:   {ANNOTATOR}
║  访问:     http://localhost:{args.port}
╠══════════════════════════════════════════════════╣
║  快捷键:  Space 播放 | ←→ 逐帧                    ║
║           S 标记起点 | E 标记终点 | Enter 确认     ║
║           Ctrl+S 保存 | Shift+←→ 跳5帧            ║
╚══════════════════════════════════════════════════╝
""")
    app.run(host="0.0.0.0", port=args.port, debug=False)
