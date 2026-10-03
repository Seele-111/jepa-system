# 可复现发行验收说明

验收日期：2026-10-03。本轮目标是让其他人取得源码后能够使用产品工作台，不是重新训练模型或提高检测评测成绩。README 中的具体开发集评测数值已经移除；原始研究记录与基线未改写。

## 分发与保留边界

- 发布两份经所有者单独授权、审核的推理定位模型副本：`models/public/locator.json`、`models/public/motion.json`，以及 registry 和 readout 许可。
- 保留原学习参数、特征顺序、模型成员权重、区间解码与拒识规则；不上传原始模型中的视频名、视频指纹、私人路径或开发数据清单。
- 真实视频/标注、私有缓存/报告、第三方源码和权重、真实本机配置不分发。公开样例由项目程序生成，不带真实异常标注或预设预测。
- 发布模型不能判断某个上传是否曾参与开发，报告以 `seen_in_development = null` 明示未知，不能将其包装成未见样本。
- 外部许可独立适用，特别是 I-JEPA 的非商业限制。项目 MIT 不构成完整模型链路的商业授权。

## 实际运行检查

| 项目 | 已验证内容 | 不代表什么 |
| --- | --- | --- |
| 干净源码目录 | 仅复制拟发布路径，未复制原模型、私人视频/标注或特征缓存 | 不是私有训练数据与历史实验的完整公开复算 |
| 新 Windows Python 环境 | 创建不继承 system-site-packages 的 Python 3.12.9 venv，仅安装 `requirements-demo.txt` | 不涵盖所有研究训练依赖 |
| 学习式 CPU | 真实 `optimized_motion` 推理；样例分析与真实 multipart 上传均产出报告及标注视频 | 不是规则回退，也不是 JEPA；不承诺检出合成变化 |
| 明确选择 CPU 规则 | `legacy --no-true-jepa` 真实产出 `cpu_motion_fallback`，结果明确标注非 JEPA | 不算优化模型或真 JEPA 验收 |
| 完整融合 | 干净源码目录生成新特征，`optimized_true_jepa_motion`；RGB、corrected 通道成功，V/I 为真实 masked predictor，`feature_cache_used=false` | 不是新独立质量评测，也不是所有硬件都能运行的证明 |
| 历史真 JEPA | 干净源码目录由新前端 venv 调用固定 WSL 模型环境，真实 `true_vjepa_ijepa` 输出及渲染成功 | 不将旧对照方案宣称为更优基线 |
| API 完整工作流 | 初次样例列表只生成媒体、预览、串行任务状态、回看、显式重跑、JSON/原视频/标注视频下载、重启后恢复完成任务 | 不是公网服务或多用户部署验收 |
| 浏览器 | 实际查看公开样例、区分历史与新推理、切换原视频/叠加视频、重新分析，H.264 播放器实际可解码 | FFmpeg 不可用时仍不能保证内嵌播放 |
| 拒绝边界 | 跨站提交被拒；私人 job 状态文件不能通过下载路径访问；不完整来源、篡改发布包、重哈希绕过均被拒 | 简单本地审核不是生产级安全认证 |

**GPU 验收边界：** 已验证 Windows 前端 + Ubuntu 24.04 WSL 的配置路线；模型解释器是既有、明确声明的固定版本环境。验证复用了分别核对 SHA 的官方预训练 checkpoint；上游源码由本轮官方固定提交资源工具取得，转换权重在本轮重新构建。没有借用私有源码补丁、训练视频、标签或特征缓存。没有把重新安装整个 CUDA/Python 环境、重新下载全部大权重、其他 GPU、原生 Linux 独立部署或普通 CPU 跑完整 JEPA 说成已经通过。

## 自动检查

新 Windows venv 的七组检查：

| 测试文件 | 执行数 | 通过 | 跳过 |
| --- | ---: | ---: | ---: |
| `test_demo.py` | 19 | 19 | 0 |
| `test_public_samples.py` | 16 | 16 | 0 |
| `test_public_release.py` | 10 | 10 | 0 |
| `test_optimized_locator.py` | 51 | 42 | 9 |
| `test_optimized_detector.py` | 24 | 24 | 0 |
| `test_optimized_model_worker.py` | 32 | 32 | 0 |
| `test_resource_preparation.py` | 29 | 29 | 0 |

总计 181 项，172 项通过，9 项因基础环境没有 sklearn 而跳过，无失败/错误。随后在已有 WSL sklearn 环境运行 portable suite，51 项全部通过，包含这 9 项 sklearn parity 检查。没有安装新的生产依赖来把可选测试变成运行前提。

运行命令见 [RUNNING.md](RUNNING.md)。基础 tests 不替代真实 GPU/媒体/浏览器检查，GPU smoke 也不替代任何新内容盲测。

## 模型一致性与历史来源限制

- 发布副本与原定位模型在两种模型、三种 FPS 的固定人工特征输入上，逐帧预测、视频证据分和区间输出完全一致；原模型文件保持不变。
- 本轮按固定官方 full checkpoint 分支与固定 Torch 版本重建 V encoder / I BF16 文件，逐张量及文件 SHA 与原资源相同。转换器使用同 basename 暂存、先验 SHA 再排他写入，不能覆盖已存在或后出现的冲突目标。
- 历史 corrected adapter 的旧 whole-file snapshot 尚未恢复；保留原历史 SHA，不以当前 collection script 的 SHA 冒充。当前 inference descriptor 对已保存训练-profile 特征记录的比对完全一致，另独立 pin 当前实现。此证据不等于找回全部历史源码。
- 配置重定位只改审核过的资源地址，不改 checkpoint SHA、RGB runtime/preprocess、特征顺序或历史来源声明。资源准备工具与实际 loader / doctor 都拒绝未完成、缺 LICENSE 或指纹不匹配的源码。
- 可用性端点区分模型包可读取、严格审核的发布包与未检查的 GPU 就绪状态。旧研究 schema-only 包不因可读取就被宣称已验证可用。

## 用户如何体验

基础工作台、公开样例、上传、学习式 CPU、曲线/区间复核与下载：按 README 快速开始。

完整融合与真 JEPA：按 [RESOURCE_SETUP.md](RESOURCE_SETUP.md) 准备固定源码、权重、模型环境与本机 TOML，先运行 doctor，再实际分析视频。资源工具不会在启动时自动大下载，也不会把缺资源的优化模式偷偷变成规则输出。

这次发行解决可分发、可配置与流程可用问题，没有宣称检测准确性因此提升；困难案例与低置信度仍需人工复核。
