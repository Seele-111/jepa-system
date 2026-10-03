#!/usr/bin/env python3
"""
JEPA 双路并行 Web V3 — V-JEPA 预测热力图 + I-JEPA 空间检测
用法: python web_app_v3.py
访问: http://localhost:5002
"""

import os, sys, json, glob, uuid, shutil, subprocess, threading
from flask import Flask, render_template_string, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = '/home/zzy/jepa_data/uploads'
JEPA_DIR = '/home/zzy/jepa_data'
RESULTS_DIR = '/home/zzy/jepa_data/web_results_v3'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

jobs = {}


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'no file'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.mp4', '.webm', '.avi', '.mov', '.mkv']:
        return jsonify({'error': f'unsupported: {ext}'}), 400

    job_id = str(uuid.uuid4())[:8]
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{job_id}_{secure_filename(file.filename)}')
    file.save(video_path)

    job = {
        'id': job_id, 'video': os.path.basename(video_path), 'path': video_path,
        'size': f'{os.path.getsize(video_path)/(1024*1024):.1f}MB',
        'status': 'extracting', 'progress': 0, 'result': None,
    }
    jobs[job_id] = job

    thread = threading.Thread(target=run_detection, args=(job_id, video_path))
    thread.daemon = True
    thread.start()
    return jsonify({'job_id': job_id, 'status': 'started'})


@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'status': job['status'], 'progress': job['progress'], 'result': job.get('result')})


@app.route('/result/<job_id>')
def result(job_id):
    job = jobs.get(job_id)
    if not job or job['status'] != 'done':
        return jsonify({'error': 'not ready'}), 404

    result_dir = os.path.join(RESULTS_DIR, job_id)
    report_path = os.path.join(result_dir, 'report.json')
    prompt_path = os.path.join(result_dir, 'llm_prompt.txt')

    report, prompt = {}, ''
    if os.path.exists(report_path):
        with open(report_path) as f: report = json.load(f)
    if os.path.exists(prompt_path):
        with open(prompt_path) as f: prompt = f.read()

    # V-JEPA heatmap frames
    vjepa_dir = os.path.join(result_dir, 'frames', 'vjepa_heatmaps')
    vjepa_frames = sorted(os.listdir(vjepa_dir)) if os.path.exists(vjepa_dir) else []

    # I-JEPA box frames
    ijepa_dir = os.path.join(result_dir, 'frames', 'ijepa_boxes')
    ijepa_frames = sorted(os.listdir(ijepa_dir)) if os.path.exists(ijepa_dir) else []

    # Combined frames
    combined_dir = os.path.join(result_dir, 'frames', 'combined')
    combined_frames = sorted(os.listdir(combined_dir)) if os.path.exists(combined_dir) else []

    # Video
    annotated_path = os.path.join(result_dir, 'annotated.mp4')
    video_url = f'/video/{job_id}/annotated.mp4' if os.path.exists(annotated_path) else None
    original_url = f'/original/{job_id}' if job.get('video') else None

    # V-JEPA timeseries
    timeseries_path = os.path.join(result_dir, 'vjepa_timeseries.json')
    timeseries = None
    if os.path.exists(timeseries_path):
        with open(timeseries_path) as f:
            timeseries = json.load(f)

    # Repair packages
    repair_packages = []
    repair_dir = os.path.join(result_dir, 'repair_packages')
    if os.path.exists(repair_dir):
        for seg_dir in sorted(os.listdir(repair_dir)):
            seg_path = os.path.join(repair_dir, seg_dir)
            if not os.path.isdir(seg_path):
                continue
            meta_path = os.path.join(seg_path, 'meta.json')
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            repair_packages.append({
                'folder': seg_dir,
                'anomalous_frames': meta.get('repair_frames', []),
                'extracted_count': meta.get('extracted_frames', 0),
                'segment': meta.get('segment', {}),
            })

    # Anomaly clips
    anomaly_clips = []
    clip_dir = os.path.join(result_dir, 'anomaly_clips')
    if os.path.exists(clip_dir):
        for fname in sorted(os.listdir(clip_dir)):
            if fname.endswith('_heatmap.mp4'):
                prefix = fname.replace('_heatmap.mp4', '')
                orig = f"{prefix}_original.mp4"
                anomaly_clips.append({
                    'prefix': prefix,
                    'heatmap_url': f'/anomaly_clips/{job_id}/{fname}',
                    'original_url': f'/anomaly_clips/{job_id}/{orig}' if os.path.exists(os.path.join(clip_dir, orig)) else None,
                })

    return jsonify({
        'report': report,
        'prompt': prompt,
        'vjepa_frames': vjepa_frames,
        'ijepa_frames': ijepa_frames,
        'combined_frames': combined_frames,
        'video_url': video_url,
        'original_video_url': original_url,
        'timeseries': timeseries,
        'has_repair_packages': job.get('has_repair_packages', False),
        'repair_packages_dir': f'repair_packages',
        'repair_packages': repair_packages,
        'anomaly_clips': anomaly_clips,
        'segment_frames': report.get('segment_frames', []),
    })


@app.route('/frames_vjepa/<job_id>/<filename>')
def serve_vjepa_frame(job_id, filename):
    return send_from_directory(os.path.join(RESULTS_DIR, job_id, 'frames', 'vjepa_heatmaps'), filename)

@app.route('/frames_ijepa/<job_id>/<filename>')
def serve_ijepa_frame(job_id, filename):
    return send_from_directory(os.path.join(RESULTS_DIR, job_id, 'frames', 'ijepa_boxes'), filename)

@app.route('/frames_combined/<job_id>/<filename>')
def serve_combined_frame(job_id, filename):
    return send_from_directory(os.path.join(RESULTS_DIR, job_id, 'frames', 'combined'), filename)

@app.route('/video/<job_id>/<filename>')
def serve_video(job_id, filename):
    result_dir = os.path.join(RESULTS_DIR, job_id)
    video_path = os.path.join(result_dir, filename)
    if not os.path.exists(video_path):
        return jsonify({'error': 'video not found'}), 404
    return send_file(video_path, mimetype='video/mp4')

@app.route('/anomaly_clips/<job_id>/<filename>')
def serve_clip(job_id, filename):
    clip_path = os.path.join(RESULTS_DIR, job_id, 'anomaly_clips', filename)
    if not os.path.exists(clip_path):
        return jsonify({'error': 'clip not found'}), 404
    return send_file(clip_path, mimetype='video/mp4')

@app.route('/segments/<job_id>/<seg_dir>/<path:filename>')
def serve_segment_frame(job_id, seg_dir, filename):
    frame_path = os.path.join(RESULTS_DIR, job_id, 'frames', 'segments', seg_dir, filename)
    if not os.path.exists(frame_path):
        return jsonify({'error': 'frame not found'}), 404
    return send_file(frame_path, mimetype='image/jpeg')

@app.route('/original/<job_id>')
def serve_original(job_id):
    job = jobs.get(job_id)
    if not job or not os.path.exists(job['path']):
        return jsonify({'error': 'original video not found'}), 404
    video_path = job['path']
    ext = os.path.splitext(video_path)[1].lower()
    mime = {'.mp4': 'video/mp4', '.webm': 'video/webm', '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime', '.mkv': 'video/x-matroska'}
    return send_file(video_path, mimetype=mime.get(ext, 'video/mp4'))

@app.route('/download/<job_id>')
def download(job_id):
    result_dir = os.path.join(RESULTS_DIR, job_id)
    zip_path = os.path.join(RESULTS_DIR, f'{job_id}.zip')
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', result_dir)
    return send_file(zip_path, as_attachment=True, download_name=f'jepa_v4_{job_id}.zip')

@app.route('/repair_packages/<job_id>/<path:filepath>')
def serve_repair_file(job_id, filepath):
    """Serve files from repair_packages (original frames, etc.)."""
    return send_from_directory(
        os.path.join(RESULTS_DIR, job_id, 'repair_packages'), filepath
    )


def run_detection(job_id, video_path):
    job = jobs[job_id]
    result_dir = os.path.join(RESULTS_DIR, job_id)
    os.makedirs(result_dir, exist_ok=True)

    try:
        job['status'] = 'extracting'
        job['progress'] = 5

        wsl_video = f'/home/zzy/jepa_data/uploads/{os.path.basename(video_path)}'
        env = {**os.environ, 'PATH': '/home/zzy/vjepa2-main/vjepa-env/bin:' + os.environ.get('PATH', '')}

        if not os.path.exists(wsl_video):
            raise FileNotFoundError(f'Video not found: {wsl_video}')

        # Phase 1: V-JEPA tokens (V-JEPA uses its own venv)
        job['progress'] = 10
        job['status_msg'] = 'Extracting V-JEPA tokens...'

        # Phase 2: Run v4 detection (all phases)
        job['progress'] = 20
        job['status_msg'] = 'Running dual-model detection...'
        result = subprocess.run([
            '/home/zzy/vjepa2-main/vjepa-env/bin/python',
            f'{JEPA_DIR}/detect_and_report_v4.py',
            '--video', wsl_video, '--output', result_dir,
            '--threshold', '0.70', '--min-gap', '2', '--min-length', '5',
            '--use-predictor-embed', '--use-optical-flow', '--use-frequency',
            '--use-depth', '--use-clip'
        ], cwd=JEPA_DIR, env=env, capture_output=True, text=True, timeout=900)

        if result.returncode != 0:
            err = f'detect_and_report_v4 failed (rc={result.returncode}): {result.stderr[-3000:]}'
            print(f'[ERROR] {err}', flush=True)
            raise RuntimeError(err)

        report_path = os.path.join(result_dir, 'report.json')
        if not os.path.exists(report_path):
            raise FileNotFoundError(f'report.json not generated')

        # Parse report
        with open(report_path) as f:
            report = json.load(f)

        stats = report.get('stats', {})
        vjepa_count = len(report.get('vjepa_anomalies', []))
        ijepa_count = len(report.get('ijepa_anomalies', []))
        fused_count = stats.get('total', 0)

        job['result'] = {
            'vjepa_segments': vjepa_count,
            'ijepa_frames': ijepa_count,
            'fused': fused_count,
            'vjepa_only': stats.get('vjepa_only', 0),
            'ijepa_only': stats.get('ijepa_only', 0),
            'both': stats.get('both', 0),
            'verdict': 'anomalous' if fused_count > 3 else 'suspicious' if fused_count > 0 else 'normal',
        }

        # Phase 3: Render annotated video
        job['progress'] = 80
        job['status_msg'] = 'Rendering dual-model video...'
        video_out = os.path.join(result_dir, 'annotated.mp4')
        result2 = subprocess.run([
            'python', f'{JEPA_DIR}/render_video_v3.py',
            '--video', wsl_video,
            '--report', report_path,
            '--output', video_out
        ], cwd=JEPA_DIR, env=env, capture_output=True, text=True, timeout=300)

        if result2.returncode != 0:
            err = f'render_video_v3 failed: {result2.stderr[-2000:]}'
            print(f'[WARN] {err}', flush=True)
            job['video_error'] = err

        # Phase 4: Extract repair packages
        job['progress'] = 95
        job['status_msg'] = 'Packaging repair frames...'
        repair_dir = os.path.join(result_dir, 'repair_packages')
        result3 = subprocess.run([
            '/home/zzy/vjepa2-main/vjepa-env/bin/python',
            f'{JEPA_DIR}/extract_repair_packages.py',
            '--report', report_path,
            '--video', wsl_video,
            '--output', repair_dir,
            '--context', '3'
        ], cwd=JEPA_DIR, env=env, capture_output=True, text=True, timeout=120)
        if result3.returncode != 0:
            print(f'[WARN] extract_repair_packages failed: {result3.stderr[-500:]}', flush=True)
        else:
            job['has_repair_packages'] = True

        job['status'] = 'done'
        job['progress'] = 100

    except Exception as e:
        import traceback as _tb
        _err = f'Job {job_id} failed: {_tb.format_exc()}'
        print(f'[ERROR] {_err}', flush=True)
        job['status'] = 'error'
        job['result'] = {'error': str(e), 'traceback': _tb.format_exc()}


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JEPA V4 双路并行异常检测</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#1a1a2e;color:#e0e0e0;min-height:100vh;display:flex;flex-direction:column;align-items:center}
.header{background:linear-gradient(135deg,#0f3460,#16213e);color:#fff;width:100%;padding:24px;text-align:center}
.header h1{font-size:20px;font-weight:500}.header p{font-size:13px;opacity:.7;margin-top:4px}
.container{max-width:960px;width:100%;padding:20px}
.upload-zone{border:2px dashed #444;border-radius:16px;padding:60px 20px;text-align:center;background:#16213e;transition:all .3s;cursor:pointer;margin:20px 0}
.upload-zone:hover,.upload-zone.drag{border-color:#e94560;background:#1a1a3e}
.upload-zone .icon{font-size:48px;margin-bottom:12px}
.upload-zone h2{font-size:16px;color:#ccc;margin-bottom:8px}
.upload-zone p{font-size:12px;color:#666}
input[type=file]{display:none}
.progress{display:none;background:#16213e;border-radius:12px;padding:20px;margin:16px 0}
.progress .bar-bg{background:#333;border-radius:8px;height:8px;overflow:hidden}
.progress .bar-fill{background:linear-gradient(90deg,#e94560,#0f3460);height:100%;width:0;transition:width .5s;border-radius:8px}
.progress .status{font-size:12px;color:#999;margin-top:10px}
.result{display:none;background:#16213e;border-radius:12px;padding:20px;margin:16px 0}
.result h3{font-size:15px;margin:16px 0 10px;color:#ccc}
.stats-row{display:flex;gap:10px;margin:10px 0}
.stat-box{flex:1;text-align:center;padding:14px 8px;background:#1a1a3e;border-radius:8px;border-left:3px solid #e94560}
.stat-box.ijepa{border-left-color:#e94560}.stat-box.vjepa{border-left-color:#3498db}.stat-box.both{border-left-color:#2ecc71}
.stat-box .v{font-size:22px;font-weight:600}.stat-box .l{font-size:10px;color:#888;margin-top:4px}
.verdict{font-size:22px;font-weight:500;text-align:center;padding:16px}
.video-section{margin:16px 0}
.video-section video{width:100%;border-radius:8px;max-height:400px;background:#000}
.video-section .video-label{font-size:12px;color:#888;margin-bottom:4px}

/* Tab switcher for model views */
.tab-bar{display:flex;gap:0;border-radius:8px;overflow:hidden;margin:12px 0}
.tab-btn{flex:1;padding:10px;background:#1a1a3e;border:none;color:#666;cursor:pointer;font-size:12px;font-weight:500;transition:all .2s}
.tab-btn.active{background:#e94560;color:#fff}.tab-btn.ijepa-a.active{background:#3498db}
.tab-btn.both-a.active{background:#2ecc71}

.frames-strip{display:flex;gap:8px;overflow-x:auto;padding:8px 0}
.frames-strip img{height:140px;border-radius:6px;flex-shrink:0;cursor:pointer;transition:transform .2s;object-fit:contain}
.frames-strip img:hover{transform:scale(1.05)}

.compare-row{display:flex;gap:8px;background:#1a1a3e;border-radius:8px;padding:8px;margin:8px 0;align-items:flex-start}
.compare-row .side{flex:1;text-align:center}
.compare-row .side .label{font-size:11px;color:#888;margin-bottom:4px}
.compare-row img{width:100%;border-radius:4px;max-height:300px;object-fit:contain}

pre{background:#0d1117;color:#c9d1d9;padding:14px;border-radius:8px;font-size:11px;max-height:250px;overflow-y:auto;white-space:pre-wrap}
.btn{display:inline-block;padding:10px 24px;border-radius:20px;font-size:13px;font-weight:500;cursor:pointer;border:none;text-decoration:none}
.btn-primary{background:#e94560;color:#fff}
.btns{display:flex;gap:10px;margin-top:12px}
.error{background:#3d1a1a;color:#e74c3c;padding:12px;border-radius:8px;font-size:12px}
.source-badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:500;margin-right:4px}
.source-vjepa{background:#3498db;color:#fff}.source-ijepa{background:#e94560;color:#fff}.source-both{background:#2ecc71;color:#000}
</style></head><body>

<div class="header">
  <h1>🔄 JEPA V4 双路并行异常检测</h1>
  <p>V-JEPA 时序预测 + I-JEPA 空间检测 · 独立打分 · 并集融合</p>
</div>

<div class="container">
<div class="upload-zone" id="dropZone">
  <div class="icon">🎬</div>
  <h2>拖拽视频到这里 或 点击上传</h2>
  <p>支持 MP4 / WebM / AVI / MOV (最大 500MB)</p>
</div>
<input type="file" id="fileInput" accept="video/*">

<div class="progress" id="progressBar">
  <div class="bar-bg"><div class="bar-fill" id="barFill"></div></div>
  <div class="status" id="progressStatus">提取特征中...</div>
</div>

<div class="result" id="resultBox">
  <div class="verdict" id="verdict"></div>
  <div class="stats-row">
    <div class="stat-box vjepa"><div class="v" id="vjepaCount" style="color:#3498db">-</div><div class="l">V-JEPA 异常段</div></div>
    <div class="stat-box ijepa"><div class="v" id="ijepaCount" style="color:#e94560">-</div><div class="l">I-JEPA 异常帧</div></div>
    <div class="stat-box both"><div class="v" id="fusedCount" style="color:#2ecc71">-</div><div class="l">融合结果</div></div>
  </div>
  <div class="stats-row">
    <div class="stat-box"><div class="v" id="vOnly" style="color:#3498db">-</div><div class="l">V-JEPA only</div></div>
    <div class="stat-box"><div class="v" id="iOnly" style="color:#e94560">-</div><div class="l">I-JEPA only</div></div>
    <div class="stat-box"><div class="v" id="bBoth" style="color:#2ecc71">-</div><div class="l">Both</div></div>
  </div>

  <!-- Annotated Video -->
  <div class="video-section" id="annotatedSection" style="display:none">
    <h3>🎬 双模型标注视频</h3>
    <video id="annotVideo" controls playsinline preload="metadata" width="100%"></video>
  </div>

  <div class="video-section" id="originalSection" style="display:none">
    <h3>📹 原始视频</h3>
    <video id="originalVideo" controls playsinline preload="metadata" width="100%"></video>
  </div>

  <!-- Tab: Switch between model views -->
  <div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('combined')">🔄 Combined</button>
    <button class="tab-btn" id="tabVjepa" onclick="switchTab('vjepa')">🔵 V-JEPA 预测热力图</button>
    <button class="tab-btn" id="tabIjepa" onclick="switchTab('ijepa')">🔴 I-JEPA 空间标注</button>
  </div>

  <h3 id="framesTitle">🔄 Combined 对比</h3>
  <div class="frames-strip" id="framesStrip"></div>

  <h3>LLM 诊断 Prompt</h3>
  <pre id="promptText"></pre>

  <h3>🔧 修复包 (Repair Packages)</h3>
  <div id="repairSection" style="min-height:40px;color:#999;font-size:13px">(检测完成后自动生成)</div>

  <h3>🎞️ 异常段视频片段 (带热力图)</h3>
  <div id="clipsSection" style="min-height:40px;color:#999;font-size:13px">(检测完成后自动生成)</div>

  <div class="btns"><a class="btn btn-primary" id="downloadBtn" href="#">下载完整报告</a></div>
</div>
</div>

<script>
var currentTab='combined',currentJobId=null;
var vjepaFrames=[],ijepaFrames=[],combinedFrames=[];

var dropZone=document.getElementById("dropZone"),fileInput=document.getElementById("fileInput"),
progressBar=document.getElementById("progressBar"),resultBox=document.getElementById("resultBox");

dropZone.addEventListener("click",function(){fileInput.click()});
dropZone.addEventListener("dragover",function(e){e.preventDefault();dropZone.classList.add("drag")});
dropZone.addEventListener("dragleave",function(){dropZone.classList.remove("drag")});
dropZone.addEventListener("drop",function(e){e.preventDefault();dropZone.classList.remove("drag");handleFile(e.dataTransfer.files[0])});
fileInput.addEventListener("change",function(e){handleFile(e.target.files[0])});

async function handleFile(file){
if(!file)return;
dropZone.style.display="none";progressBar.style.display="block";resultBox.style.display="none";
var form=new FormData();form.append("video",file);
document.getElementById("progressStatus").textContent="Uploading...";
document.getElementById("barFill").style.width="5%";
var resp=await fetch("/upload",{method:"POST",body:form});var data=await resp.json();
if(data.error){showError(data.error);return}
document.getElementById("progressStatus").textContent="Extracting V-JEPA & I-JEPA features...";
var jobId=data.job_id;
while(true){
  await new Promise(function(r){setTimeout(r,2000)});
  var s=await fetch("/status/"+jobId);var status=await s.json();
  if(status.status==="error"){showError(status.result&&status.result.error||"unknown");return}
  if(status.status==="done"){showResult(jobId);return}
  var pct=status.progress||10;
  document.getElementById("barFill").style.width=pct+"%";
  document.getElementById("progressStatus").textContent=status.status_msg||"Processing...";
}}

async function showResult(jobId){
currentJobId=jobId;
progressBar.style.display="none";resultBox.style.display="block";
var r=await fetch("/result/"+jobId);var data=await r.json();var rep=data.report||{};

var stats=rep.stats||{};
var vjepaAnoms=rep.vjepa_anomalies||[],ijepaAnoms=rep.ijepa_anomalies||[];
var fused=rep.fused_anomalies||[];

document.getElementById("verdict").textContent=
  fused.length>3?"🔴 异常":fused.length>0?"🟡 可疑":"🟢 正常";
document.getElementById("vjepaCount").textContent=vjepaAnoms.length;
document.getElementById("ijepaCount").textContent=ijepaAnoms.length;
document.getElementById("fusedCount").textContent=fused.length;
document.getElementById("vOnly").textContent=stats.vjepa_only||0;
document.getElementById("iOnly").textContent=stats.ijepa_only||0;
document.getElementById("bBoth").textContent=stats.both||0;

// Videos
if(data.video_url){
  document.getElementById("annotatedSection").style.display="block";
  document.getElementById("annotVideo").src=data.video_url;
}
if(data.original_video_url){
  document.getElementById("originalSection").style.display="block";
  document.getElementById("originalVideo").src=data.original_video_url;
}

document.getElementById("promptText").textContent=data.prompt||"(none)";
document.getElementById("downloadBtn").href="/download/"+jobId;

// Frames
vjepaFrames=data.vjepa_frames||[];
ijepaFrames=data.ijepa_frames||[];
combinedFrames=data.combined_frames||[];
switchTab(currentTab);  // redraw

// Repair packages
var rpDiv=document.getElementById("repairSection");
if(data.repair_packages&&data.repair_packages.length>0){
  rpDiv.innerHTML="";
  for(var i=0;i<data.repair_packages.length;i++){
    var pkg=data.repair_packages[i];
    var card=document.createElement("div");
    card.style.cssText="background:#1a1a3e;border-radius:8px;padding:12px;margin:8px 0;border-left:3px solid #2ecc71";
    
    var title=document.createElement("div");
    title.style.cssText="font-size:14px;font-weight:500;color:#2ecc71;margin-bottom:6px";
    title.textContent="📦 "+pkg.folder;
    card.appendChild(title);
    
    var info=document.createElement("div");
    info.style.cssText="font-size:11px;color:#888";
    info.innerHTML="异常帧: "+pkg.anomalous_frames.join(", ").substring(0,80)+(pkg.anomalous_frames.length>5?"...":"")+" | 总帧: "+pkg.extracted_count;
    card.appendChild(info);
    
    var actions=document.createElement("div");
    actions.style.cssText="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap";
    
    // Download original frames
    var dlLink=document.createElement("a");
    dlLink.href="/repair_packages/"+jobId+"/"+encodeURIComponent(pkg.folder)+"/repair_prompt.txt";
    dlLink.style.cssText="font-size:11px;color:#3498db;text-decoration:none;padding:4px 10px;background:#16213e;border-radius:4px";
    dlLink.textContent="📋 repair prompt";
    dlLink.target="_blank";
    actions.appendChild(dlLink);
    
    // Download original frame preview
    var firstFrame=pkg.anomalous_frames[0];
    if(firstFrame!==undefined){
      var fname="frame_"+("00000"+firstFrame).slice(-5)+".png";
      var imgLink=document.createElement("a");
      imgLink.href="/repair_packages/"+jobId+"/"+encodeURIComponent(pkg.folder)+"/original/"+fname;
      imgLink.style.cssText="font-size:11px;color:#e94560;text-decoration:none;padding:4px 10px;background:#16213e;border-radius:4px";
      imgLink.textContent="🖼 view frame "+firstFrame;
      imgLink.target="_blank";
      actions.appendChild(imgLink);
    }
    
    card.appendChild(actions);
    rpDiv.appendChild(card);
  }
}else{
  rpDiv.innerHTML='<span style="color:#666">(无修复包)</span>';
}

// Anomaly clips with expandable frames
// Anomaly clips with expandable frames (heatmap + original)
var clipsDiv=document.getElementById("clipsSection");
var segFrames=data.segment_frames||[];
if(data.anomaly_clips&&data.anomaly_clips.length>0){
  clipsDiv.innerHTML="";
  for(var i=0;i<data.anomaly_clips.length;i++){
    var clip=data.anomaly_clips[i];
    var seg=segFrames[i]||{};
    var card=document.createElement("div");
    card.style.cssText="background:#1a1a3e;border-radius:8px;padding:12px;margin:8px 0;border-left:3px solid #f39c12";
    var title=document.createElement("div");
    title.style.cssText="font-size:14px;font-weight:500;color:#f39c12;margin-bottom:6px;cursor:pointer";
    title.textContent="🎞️ "+clip.prefix+" ("+(seg.frame_count||0)+"帧)";
    card.appendChild(title);
    var vid=document.createElement("video");
    vid.src=clip.heatmap_url;
    vid.controls=true;vid.playsInline=true;vid.preload="metadata";
    vid.style.cssText="width:100%;max-width:720px;border-radius:6px;background:#000;margin-bottom:8px";
    card.appendChild(vid);
    // Buttons row
    var btns=document.createElement("div");
    btns.style.cssText="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px";
    // Heatmap frames toggle
    if(seg.frames&&seg.frames.length>0){
      var t1=document.createElement("button");
      t1.textContent="🔥 热力图帧 ("+seg.frames.length+")";
      t1.style.cssText="padding:4px 12px;background:#16213e;border:1px solid #e94560;border-radius:4px;color:#e94560;font-size:11px;cursor:pointer";
      (function(seg,frames,jobId,card,t1){
        t1.onclick=function(){
          var sid="strip_hm_"+seg.segment_index;
          var strip=document.getElementById(sid);
          if(strip){strip.style.display=strip.style.display==="none"?"flex":"none";return}
          var s=document.createElement("div");
          s.id=sid;s.style.cssText="display:flex;gap:4px;overflow-x:auto;padding:8px 0;flex-wrap:wrap";
          for(var j=0;j<frames.length;j++){
            var img=document.createElement("img");
            img.src="/segments/"+jobId+"/"+seg.dir+"/"+frames[j];
            img.loading="lazy";img.title=frames[j];
            img.style.cssText="height:80px;border-radius:3px;flex-shrink:0;cursor:pointer";
            img.onclick=function(src){return function(){window.open(src,"_blank")}}(img.src);
            s.appendChild(img);
          }
          card.appendChild(s);
        }
      })(seg,seg.frames,currentJobId,card,t1);
      btns.appendChild(t1);
    }
    // Original frames toggle (for ComfyUI re-generation)
    if(seg.original_frames&&seg.original_frames.length>0){
      var t2=document.createElement("button");
      t2.textContent="📥 原始帧 ("+seg.original_frames.length+")";
      t2.style.cssText="padding:4px 12px;background:#16213e;border:1px solid #2ecc71;border-radius:4px;color:#2ecc71;font-size:11px;cursor:pointer";
      (function(seg,frames,jobId,card,t2){
        t2.onclick=function(){
          var sid="strip_orig_"+seg.segment_index;
          var strip=document.getElementById(sid);
          if(strip){strip.style.display=strip.style.display==="none"?"flex":"none";return}
          var s=document.createElement("div");
          s.id=sid;s.style.cssText="display:flex;gap:4px;overflow-x:auto;padding:8px 0;flex-wrap:wrap";
          for(var j=0;j<frames.length;j++){
            var img=document.createElement("img");
            img.src="/segments/"+jobId+"/"+seg.dir+"/"+frames[j];
            img.loading="lazy";img.title=frames[j];
            img.style.cssText="height:80px;border-radius:3px;flex-shrink:0;cursor:pointer";
            img.onclick=function(src){return function(){window.open(src,"_blank")}}(img.src);
            s.appendChild(img);
          }
          card.appendChild(s);
        }
      })(seg,seg.original_frames,currentJobId,card,t2);
      btns.appendChild(t2);
    }
    card.appendChild(btns);
    clipsDiv.appendChild(card);
  }
}else{
  clipsDiv.innerHTML='<span style="color:#666">(无异常片段)</span>';
}


}

function switchTab(tab){
currentTab=tab;
var btns=document.querySelectorAll(".tab-btn");btns.forEach(function(b){b.classList.remove("active")});
document.getElementById("tab"+(tab==="combined"?"":" ").split(" ")[0]);
if(tab==="vjepa")document.querySelector(".tab-btn:nth-child(2)").classList.add("active");
else if(tab==="ijepa")document.querySelector(".tab-btn:nth-child(3)").classList.add("active");
else document.querySelector(".tab-btn:nth-child(1)").classList.add("active");

var title=document.getElementById("framesTitle");
var strip=document.getElementById("framesStrip");strip.innerHTML="";

var frames=[],prefix="";
if(tab==="combined"){frames=combinedFrames;prefix="frames_combined";title.textContent="🔄 Combined 对比"}
else if(tab==="vjepa"){frames=vjepaFrames;prefix="frames_vjepa";title.textContent="🔵 V-JEPA 预测热力图 (蓝=预测准确 红=意外)"}
else{frames=ijepaFrames;prefix="frames_ijepa";title.textContent="🔴 I-JEPA 空间异常框 (红/橙=高异常)"}

if(frames.length===0){strip.innerHTML='<p style="color:#666;font-size:13px">(无标注帧)</p>';return}
for(var i=0;i<frames.length;i++){
  var img=document.createElement("img");
  img.src="/"+prefix+"/"+currentJobId+"/"+frames[i];
  img.loading="lazy";img.title=frames[i];
  strip.appendChild(img);
}
}

function showError(msg){
progressBar.style.display="none";resultBox.style.display="block";
document.getElementById("verdict").innerHTML='<div class="error">'+msg+'</div>';
}
</script></body></html>'''


if __name__ == '__main__':
    print("JEPA V4 双路并行 Web (V-JEPA + I-JEPA)")
    print("http://localhost:5002")
    app.run(host='0.0.0.0', port=5002, debug=False)
