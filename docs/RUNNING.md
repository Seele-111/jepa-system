# 运行指南

## 1. 下载后直接体验（学习式 CPU）

本次发行包含下游模型与公开样例生成器。基础依赖足以运行 `optimized_fast`、上传分析、证据曲线、区间回看、视频与 JSON 下载；不需要开发者的私有视频、缓存或大模型环境。

Windows / PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe code\demo_app.py
```

Linux：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-demo.txt
.venv/bin/python code/demo_app.py
```

打开 `http://127.0.0.1:5002/`。首次载入样例列表只生成视频，不运行推理；先预览，再点击当前模式分析。Synthetic 样例没有真实标注，也不保证某个变化能被检出。

建议安装 FFmpeg，以输出浏览器兼容的 H.264。无 FFmpeg 仍输出可下载的视频，但内嵌播放可能失败；标注片不保留音轨。

## 2. 模式选择与命令行

| 模式 | 含义 | 资源 |
| --- | --- | --- |
| `optimized_fast` | 学习式 CPU 运动模型，不是 JEPA，不是规则回退 | 发布的 motion readout + 基础依赖 |
| `optimized` | 当前完整 JEPA / 运动 / RGB 融合基线 | 发布的 locator readout + 配置的上游源码、权重、CUDA |
| `legacy` | 历史真 JEPA 对照；失败时明确报告 CPU fallback | 同一组 V/I 资源；规则回退不需要大模型 |

命令行学习式 CPU：

```powershell
python code\generate_public_samples.py
python code\demo_detector.py --video output\public-samples\synthetic_jump.mp4 --output output\my-cpu-run --algorithm optimized_fast
```

显式规则 CPU（与学习式模式不同）：

```powershell
python code\demo_detector.py --video output\public-samples\synthetic_jump.mp4 --output output\my-rules-run --algorithm legacy --no-true-jepa
```

工作台选择旧对照模式后，也可在“旧对照方式”中显式选择 CPU 规则。报告的 `method` 和 `jepa_usage` 必须与真实路径一致；不能用 fallback 充当真 JEPA 验证。

## 3. 完整优化 / 真 JEPA

先完成 [资源准备指南](RESOURCE_SETUP.md)，再将示例配置复制为 `jepa-runtime.toml`，或者设置 `JEPA_CONFIG` 指向自己的配置文件。真实配置不参与 Git 发布。

```powershell
$env:JEPA_CONFIG = "C:\path\to\jepa-runtime.toml"
python code\model_doctor.py                  # 只检查发布模型，不能代表 GPU 已就绪
python code\model_doctor.py --model-runtime  # 在配置的解释器中核查源码、checkpoint SHA、版本和 CUDA
python code\demo_app.py
```

完整新视频分析：

```powershell
python code\demo_detector.py --video output\public-samples\synthetic_jump.mp4 --output output\my-full-run --algorithm optimized
python code\demo_detector.py --video output\public-samples\synthetic_jump.mp4 --output output\my-legacy-run --algorithm legacy
```

Linux 可直接使用原生 model interpreter；Windows 可以把前端放在 Windows、模型放在 WSL。配置里的 `resources` 路径由模型环境解释，WSL 请使用 Linux 路径；相对路径基于项目 checkout。优化脚本不再默认使用开发者的解释器或私人模型目录。

### 可选热 Worker

在同一个配置与模型环境中运行：

```bash
python -u -B code/optimized_model_worker.py --config /path/to/jepa-runtime.toml --serve
```

默认是本机 5004，可通过 `runtime.worker_port` 调整。前端只接受代码 SHA、项目根与配置身份一致的 Worker，否则启动新的单次进程，不借用未知服务或缓存的视频结果。

`runtime.retain_models = true` 保留热模型，连续处理更快；设为 `false` 会在每个通道结束后释放模型，降低显存占用但增加再次加载耗时。两者使用相同的特征与学习参数，且保留原 mask seed / counter 隔离策略。

## 4. 输入、产物与来源

- Web 上传上限为 500 MB、120 秒、10000 帧，任务串行执行，排队有上限。
- 新任务写入 `output/demo-runs/`，媒体、报告、真实配置都被 Git 忽略。
- CLI 完整 GPU 输入应放在项目 `output/`，或通过 `demo.data_root` / `JEPA_DEMO_DATA` 指定额外本地输入目录；profile snapshot 必须位于项目输出根。不要放宽 Worker 的路径边界。
- 帧区间是 **start/end inclusive**；秒区间是 **start inclusive / end exclusive**。FPS 来自源视频，不靠猜测补齐。
- `recorded_demo`、新推理和 `saved_inference` 保持区分。公开样例的回看结果只来自真实完成任务，不预置“正确结果”。
- 若拥有旧 0207/0317 原始开发样例与报告，可以继续通过原入口使用；它们不随发行提供。当前合成样例不冒充原始案例。
- 公开模型不分发开发视频指纹，报告中训练成员判断为未知，而不是谎称所有上传均未参与开发。
- HTTP `/health` 只证明服务在线；`/api/capabilities` 区分模型包存在与尚未检查的外部资源。

## 5. 检查与排障

```text
python -m unittest discover -s code -p "test_demo.py" -v
python -m unittest discover -s code -p "test_public_samples.py" -v
python -m unittest discover -s code -p "test_public_release.py" -v
python -m unittest discover -s code -p "test_optimized_locator.py" -v
python -m unittest discover -s code -p "test_optimized_detector.py" -v
python -m unittest discover -s code -p "test_optimized_model_worker.py" -v
```

缺少 sklearn 时，portable sklearn parity 会明确跳过。历史训练、其他研究分支和全库测试有额外数据/依赖，基础 requirements 并不承诺覆盖它们。

| 现象 | 处理 |
| --- | --- |
| 完整优化报缺资源 | 运行 model doctor，检查配置的模型解释器与资源指南；不要更改 JSON 校验字段 |
| Torch / Python / RGB profile 不匹配 | 使用固定 model-runtime 版本，不能把任意新版本当成已验证等价 |
| 显存不足 | 关闭自己不需要的 GPU 程序，或使用 `retain_models=false`；不要把 CPU fallback 当优化结果 |
| 视频能下载、不能播放 | 检查 FFmpeg 与报告中的 video_encoding 警告 |
| 没有候选区间 | 当前模型未保留候选，不是“正常认证”；人工复核 |
| 样例生成失败 | 显式运行生成器，检查文件权限与视频编码器 |

服务仅监听 localhost。可通过 `JEPA_DEMO_PORT` 调整前端本机端口，但不要改为公网部署；本项目没有附带生产级鉴权、多用户隔离或托管服务。
