# JEPA Lens · 视频异常片段定位

一个研究 **AI 生成视频中的视觉与物理异常时间区间** 的开源项目，配套本地交互式工作台。

上传视频后，可查看候选异常区间、逐帧证据曲线、原片与标注片对照，以及下载 JSON 报告。它帮助发现值得复核的片段，不代替人工判断，也不把空候选当作“视频已经合格”。

> **研究原型。** 仓库包含可直接使用的学习式 CPU 定位模型、完整优化定位模型的发布版 readout，以及公开样例生成器。完整优化与真 JEPA 需要另外准备匹配的官方源码、预训练权重和 GPU 环境；这些资源不由项目的 MIT 许可统一授权，也不会在启动时自动下载。

## 可以体验什么

- **上传分析**：选择自己的视频，查看排队、进度与实际使用的算法。
- **候选区间与证据曲线**：查看时间范围，点击区间跳转回看。
- **原片 / 标注片对照**：在浏览器内切换，下载视频与报告。
- **公开样例**：程序生成平滑运动、位置跳变、消失与重现的视频；先预览，再用选择的模型真实分析。
- **结果回看**：已完成任务保存在本地，页面刷新或服务重启后可以重新查看。
- **三种检测模式**：学习式 CPU 运动、完整融合优化、历史真 JEPA 对照；另保留明确标注的规则 CPU 回退。

### 算法模式

| 模式 | 实际算法 | 使用条件 |
| --- | --- | --- |
| `optimized_fast` | 学习式运动定位；**不是 JEPA，也不是规则回退** | 安装基础依赖后即可运行，模型随仓库提供 |
| `optimized` | 校正 V/I-JEPA 与运动 RF/ET，加 RGB 时序头，再进行区间解码与拒识 | 发布 readout 已提供；需要按指南准备外部资源和 CUDA |
| `legacy` | 历史真 V/I-JEPA 对照 | 同样需要真实模型资源；缺资源时明确标记 CPU 回退，也可手动选择规则 CPU |

发布模型保留现有基线的学习参数、特征顺序与解码规则，不是为了开源另换了一套算法。新克隆默认选择容易启动的学习式 CPU 模式；**它不等于你配置完整资源后得到的融合模式**。目前默认也不代表所有实验中的单项最高值或已经证明最优。

## 快速开始

### Windows / PowerShell

安装 Python 3.12 后，在项目目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe code\demo_app.py
```

打开 **http://127.0.0.1:5002/**。点击一个公开样例，预览后选择“用当前模式分析样例”；也可以上传自己的视频。默认是学习式 CPU 模式，无需你的私有数据、缓存或 GPU。

### Linux

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-demo.txt
.venv/bin/python code/demo_app.py
```

安装 FFmpeg 可以输出更适合浏览器播放的 H.264。没有 FFmpeg 时仍可下载标注视频，但不能保证内嵌播放。标注视频不保留音轨。

### 获取项目

在 GitHub 点击 **Code → Download ZIP**，解压后按上述步骤运行；开发者也可以：

```text
git clone https://github.com/Seele-111/jepa-system.git
cd jepa-system
```

## 使用完整优化和真 JEPA

按照 [资源准备指南](docs/RESOURCE_SETUP.md) 取得固定版本的官方源码与权重，再复制 `jepa-runtime.example.toml` 为本地 `jepa-runtime.toml`，填入自己的资源目录、解释器与 native / WSL 设置。配置和下载资源不会加入 Git。

完整步骤、命令行分析、低显存模式、输入限制和排障见 [运行指南](docs/RUNNING.md)。模型的来源与严格校验机制见 [模型说明](models/README.md)；实际验证范围见 [发布验收说明](docs/REPRODUCIBLE_RELEASE.md)。

**资源准备不是可省略步骤。** 缺少权重、源码不匹配或 profile 校验失败时，优化模式会报错，不会偷偷用规则模型替代。原始私有训练视频、标注、缓存和录制演示不随仓库提供，也不是新用户体验工作台的前置条件。

## 能力边界

- 这是上传后使用前后文的离线定位，不承诺实时分析或检出所有物理错误。
- 分数是模型证据，不是独立校准的异常概率；低置信度和困难案例需要人工复核。
- 既有开发数据已经多轮用于研究，不作为新的独立盲测。公开 Synthetic 样例只用于体验流程，不是评测或质量证明。
- 本轮改善的是可分发性与可复现使用，不宣称检测能力因此提升。原始实验记录保持不变。
- 服务仅监听本机，不附带公网生产级鉴权、存储隔离或多用户部署。
- 第三方许可单独适用，尤其 I-JEPA 的非商用条款；不能因为本项目使用 MIT 就推定完整链路可以商用。

## 仓库与开发

```text
code/                         检测、特征、训练、评估、工作台与测试
scripts/                      既有启动和研究脚本
models/public/                经审查的下游 readout 与完整性注册表
jepa-runtime.example.toml     本机配置模板，不含私人路径
requirements-demo.txt         工作台和学习式 CPU 路径依赖
docs/                         运行、资源、发布验证与项目状态
```

欢迎使用、反馈问题和继续开发。提交问题时请给出所选模式、错误代码与运行环境，不上传敏感视频或凭据。贡献步骤见 [CONTRIBUTING.md](CONTRIBUTING.md)；修改推理、接口或评估协议前请先读 [AGENTS.md](AGENTS.md)。

## License

项目所有者有权授权的自有代码与本次发布的下游 readout 采用 [MIT License](LICENSE)。外部预训练模型、上游源码和数据遵循各自条款，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

**English summary:** JEPA Lens is an open-source research prototype for localizing candidate visual/physical anomalies in generated videos. The checkout includes CPU learned-motion readouts, a full-fusion readout, synthetic sample generation and a localhost review UI. Full fusion and genuine JEPA require separately acquired, pinned upstream resources. Results need human review; synthetic demos are not benchmarks.
