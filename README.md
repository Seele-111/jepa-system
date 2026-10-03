# JEPA Lens · 视频异常片段定位

用于研究 **AI 生成视频中的视觉/物理异常时间区间**，并提供一个本地交互式演示工作台。

输入视频后，系统输出候选异常区间、逐帧证据曲线、标注视频与 JSON 报告。候选结果需要人工复核；空候选不等于视频已经通过质量验收。

> **状态：研究原型 / 源码版。** 本仓库不包含私有视频、标注、预训练权重、当前训练模型包或已录制演示结果。新克隆可以启动前端、运行显式 CPU 运动回退流程和相应测试；不能仅凭本仓库复现当前优化模型的完整推理或开发集成绩。

## 项目做什么

- 探索 V-JEPA / I-JEPA 预测证据、局部运动残差与 RGB 时序特征在异常定位上的作用。
- 通过内容分组、折外预测和事件 IoU 指标检查定位能力，而不只看视频级分数。
- 比较候选生成、拒识、区间合并、边界细化和不同融合方案。
- 在 `127.0.0.1` 提供上传、任务状态、时间轴回看和结果下载；不作为公网生产服务。

### 当前算法与名称边界

| 模式 | 实际用途 | 额外资源 |
| --- | --- | --- |
| `optimized` | 当前本地默认：校正 JEPA + 运动 RF 0.4 / ET 0.4，RGB + 运动 TCN 0.2，再做区间解码与拒识 | 原始模型包、对应 GPU 特征环境和预训练权重 |
| `optimized_fast` | 学习式运动定位，不是 JEPA | 独立的运动模型包 |
| `legacy` | 历史真 JEPA 演示路径，失败时明确标为 CPU 运动回退 | 真 JEPA 需要配置 WSL 与外部资源；显式 CPU 回退不需要 |

“当前默认”不是“所有实验中每个指标最高”，也不代表已证明最优。未通过统一升级护栏的实验分支没有替换默认模型。CPU 运动回退不能被展示成真 JEPA 或优化算法成绩。

## 快速开始

可以在 GitHub 页面点击 **Code → Download ZIP** 下载并解压源码；已安装 Git 的开发者也可以执行：

```text
git clone https://github.com/Seele-111/jepa-system.git
cd jepa-system
```

已验证的本地前端环境：Windows、Python 3.12.9。以下命令在项目根目录的 PowerShell 中执行；仅安装前端和 CPU 路径已有依赖，不安装大型 GPU 框架。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe code\demo_app.py
```

打开 `http://127.0.0.1:5002/`。首次克隆没有模型包，页面默认的 `optimized` 模式不能完成推理；不要把页面成功打开当成算法已经可用。快捷样例也需要本地视频与录制报告，本仓库不附带它们。

### 不依赖私有模型的 CPU 流程

将 `C:\path\to\input.mp4` 替换成自己有使用权限的视频：

```powershell
.\.venv\Scripts\python.exe code\demo_detector.py `
  --video "C:\path\to\input.mp4" `
  --output "output\cpu-demo" `
  --algorithm legacy --no-true-jepa
```

该命令显式运行 **CPU motion fallback**，输出 `demo_report.json` 和 `annotated.mp4`。若本机有 FFmpeg，会尝试输出浏览器兼容的 H.264；没有时可能只能下载标注视频，不能保证浏览器播放。输出视频不保留音频。

恢复已有优化模型、配置 WSL、了解固定路径限制和常见失败，请阅读 [运行指南](docs/RUNNING.md) 与 [模型资源说明](models/README.md)。完整 GPU 路径不是跨机器一键启动包。

## 当前评测与限制

当前默认控制的内容分组折外评测：

| 指标 | 结果 |
| --- | ---: |
| 事件 F1 @ IoU 0.3 | 55.71% |
| 事件 F1 @ IoU 0.5 | 34.29% |
| Frame F1 | 50.89% |
| 正常视频产生候选 | 1 / 14 |
| 异常视频无候选 | 22 / 63 |
| 正确匹配事件 @ IoU 0.5 | 24 / 85 |

这些是 **定位指标，不是视频准确率**。开发集为 77 条视频、76 个内容 SHA 组、6133 帧和 85 个异常事件；该集已经多轮用于开发，不能再称为独立盲测。最新局部时空交互诊断没有通过预先固定的训练准入条件，未训练新定位头，也没有证明产品能力提升。

本仓库提供实现与状态摘要，不提供原始数据及全部实验产物。需要这些资源才能独立复算数值，不能把源码存在视为完整复现证据。详见 [项目状态与评测边界](docs/PROJECT_STATUS.md)。

## 仓库结构

```text
.
├── code/                     # 检测、特征、训练、评估和现有测试
│   ├── demo_app.py           # 本地 Flask 工作台
│   ├── demo_detector.py      # 演示输出与显式回退
│   ├── optimized_detector.py
│   ├── optimized_locator.py # portable 推理与事件解码
│   ├── optimized_model_worker.py
│   └── templates/demo.html
├── scripts/                  # 已有启动及研究批处理脚本
├── docs/                     # 精简运行/状态/仓库规范说明
├── models/README.md          # 所需模型的说明，不含模型本身
├── requirements-demo.txt     # 仅前端/CPU 的已验证依赖
├── AGENTS.md                 # 本仓库协作约束
├── CONTRIBUTING.md
└── THIRD_PARTY_NOTICES.md
```

保留已有代码结构以避免改变导入与算法接口。研究脚本中仍有原开发环境的绝对路径；它们不是全部可移植的安装入口，不要批量执行 `run_*` 脚本。

## 验证与参与

前端与回退的聚焦检查（使用标准库 `unittest`）：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s code -p "test_demo.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s code -p "test_optimized_locator.py" -v
```

第二条包含 portable 解码与数值测试；与 scikit-learn 的一致性检查在缺少 scikit-learn 时会明确跳过，不代表它们通过。全量训练/GPU 测试需要独立配置 PyTorch、torchvision、scikit-learn 和对应研究资源，`requirements-demo.txt` 不覆盖它们。

修改前阅读 [贡献说明](CONTRIBUTING.md)。问题反馈请提供所用模式、代码版本、错误码、资源是否齐备及不含隐私的最小复现；不要上传私有视频、完整请求体、凭据或原始日志。仓库由项目所有者维护，维护者信息以 GitHub 仓库记录为准。

## 资源、许可与发布规范

- 发布采用源码白名单，不上传私有数据、运行缓存、模型包、媒体和旧研究报告；这些资源保留在本地。
- 本仓库中项目所有者有权授权的自有代码采用 [MIT License](LICENSE)，允许使用、修改、分享和商业使用，须保留版权与许可声明。第三方资源不因此取得 MIT 授权。
- 第三方源码、权重与数据各自有独立许可；未完整审定的本地源码副本不随仓库分发。见 [第三方资源说明](THIRD_PARTY_NOTICES.md)。
- README 的内容组织、相对链接、忽略规则与资源边界参考 GitHub 官方文档；来源列在 [仓库整理依据](docs/REPOSITORY_GUIDE.md)。没有附加自动部署、Actions 工作流、Release 或生产服务。
