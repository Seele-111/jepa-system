# 运行指南

## 1. 源码版能做什么

仅凭本仓库可启动本地页面，运行显式 CPU 运动回退并检查报告/时间区间/API。当前优化算法和已录制样例所需的数据与模型不包含在仓库中，缺少它们时不能宣称完整算法已经就绪。

已核验的前端环境为 Windows / Python 3.12.9，包版本见 `requirements-demo.txt`。这是已有本地环境记录，不是所有平台安装兼容性的承诺。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe code\demo_app.py
```

页面与健康检查分别为 `http://127.0.0.1:5002/` 和 `http://127.0.0.1:5002/health`。`health` 只证明 HTTP 服务可用，不证明模型已加载或推理可用。页面默认 `optimized`，新克隆缺模型时此模式会失败，不会偷偷把 CPU 回退包装成优化推理。

`scripts/start_demo.ps1` 会使用当前 PATH 的 Python 并检查既有依赖，不会自动安装；使用隔离环境时优先直接调用上述解释器。

## 2. 显式 CPU 运动回退

```powershell
.\.venv\Scripts\python.exe code\demo_detector.py `
  --video "C:\path\to\input.mp4" --output "output\cpu-demo" `
  --algorithm legacy --no-true-jepa
```

这是 `cpu_motion_fallback`，不是 `optimized_fast` 学习模型，也不是真 JEPA。视频需可由 OpenCV 解码。输出包含 `demo_report.json` 与 `annotated.mp4`；报告记录实际方法和警告。

- 帧区间是起止帧均包含；秒区间是 start-inclusive / end-exclusive。
- 空候选不是“正常保证”；分数不是校准的物理错误概率。
- FFmpeg 为可选外部程序，本轮不自动安装；可用时尝试 H.264 编码，否则可能只保证下载而不保证浏览器播放。
- 标注视频不保留音频。

## 3. 恢复优化模型和 GPU 特征服务

首先恢复自己有权限使用的、**未经改写元数据**的原始资源：

- `models/optimized_locator_v1.json`：默认融合模型包。
- `models/optimized_motion_locator_v1.json`：学习式 CPU 运动模型包。
- 对应的 V-JEPA 编码器/预测器、I-JEPA 与 RGB R3D checkpoint；它们不由本仓库下载或分发。
- 匹配训练 profile 的 PyTorch / torchvision / CUDA 环境、预处理与特征代码。

完整默认算法通过 `code/optimized_model_worker.py` 的本地串行服务提取 GPU 特征。服务启动后 lazy 加载模型，初始健康检查成功不代表权重已经加载。在正确配置的 WSL 环境、项目根目录启动：

```text
python -B code/optimized_model_worker.py --serve
```

监听 `127.0.0.1:5004`。该 Worker 与 `optimized_detector.py` 保留原开发环境的固定发行版、解释器、输入/输出根目录和 checkpoint 路径；读者须检查实际源码。当前版本不是任意机器的通用部署配置。修改这些位置需要保持目录校验、checkpoint SHA、RGB profile、feature order 和模型 provenance 一致，并重新完成兼容性及真实视频验证；不要仅修改 JSON 让校验“通过”。

历史 `legacy` 真 JEPA 路径可通过 `JEPA_WSL_DISTRO`、`JEPA_WSL_PYTHON` 与 `JEPA_WSL_PIPELINE` 配置；这些变量不代表已解除优化 Worker 的固定路径约束。

## 4. 私有样例、上传与输出

- `JEPA_DEMO_DATA` 只配置前端演示样例的数据目录；不是优化 Worker 的输入许可根目录配置。
- 录制样例需要相应视频和已有报告。源码版中不存在它们，快捷样例不可直接复现。
- 新任务写入 `output/demo-runs/`，该目录不参与 Git 发布。
- Web 上传限制为 500 MB、120 秒和 10000 帧，并使用串行任务队列。这些是前端限制，不是算法所有脚本的统一限制。
- 服务仅供 localhost 使用，不要直接改成公网监听；没有附带生产级鉴权、存储隔离或多用户方案。

## 5. 检查与排障

```text
python -m unittest discover -s code -p "test_demo.py" -v
python -m unittest discover -s code -p "test_optimized_locator.py" -v
```

portable 测试中的 sklearn parity 在缺少 sklearn 时明确跳过。`test_optimized_detector.py` 中真实模型包相关测试需要本地恢复两个 JSON 包；源码版不能直接运行这部分。研究测试还可能需要 GPU、torchvision、scikit-learn、特征缓存和额外历史依赖；最小 requirements 不覆盖它们。

常见问题：

| 现象 | 检查项 |
| --- | --- |
| 页面已开但优化模式报错 | 模型包、WSL 解释器、5004 Worker、checkpoint/profile 校验；前端 health 不能替代推理检查 |
| 快捷样例不可用 | 本地私有样例视频与录制报告未恢复 |
| 报告标为 CPU fallback | 当前确实走了回退；不要当成真 JEPA 结果 |
| 视频可下载但不能网页播放 | FFmpeg/H.264 是否可用以及报告中的编码警告 |
| 空候选 | 当前规则未保留区间，不是正常认证；仍须人工复核 |
