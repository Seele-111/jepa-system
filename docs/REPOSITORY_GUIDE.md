# 仓库整理依据与发布边界

整理日期：2026-10-03。本次采用 GitHub 官方文档中 README 的常见内容组织：项目作用、用途、开始使用、帮助与维护信息；详细操作放入独立文档，仓库内文件通过相对链接连接。

## 已实际查阅的官方来源

以下页面在原源码整理时通过只读 HTTP 获取原文，并用于核对发布格式和边界：

- [About the repository README file](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)
- [Ignoring files](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files)
- [About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
- [Licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
- [Setting your commit email address](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address)
- [Email addresses reference](https://docs.github.com/en/account-and-profile/reference/email-addresses-reference)
- [MIT License / Choose a License](https://choosealicense.com/licenses/mit/)；GitHub 官方模板用于根层 `LICENSE`。

这些是格式和操作依据，不表示 GitHub/OpenAI 审核或认可本项目及其效果，也不存在“套用格式就达到算法质量标准”的含义。

本轮可复现发行另核对了固定提交的 V-JEPA/I-JEPA 官方 README 与 LICENSE、Python 3.12.3 官方发布页、Torch/torchvision 官方 cu132 wheel 索引及 torchvision R3D 定义。资源名称、许可和执行步骤见 [资源准备指南](RESOURCE_SETUP.md) 与 [第三方说明](../THIRD_PARTY_NOTICES.md)，不采用未固定的 main/latest 或整个上游训练环境。

## 纳入的内容

- 自有 Python/前端源码、现有测试和研究/启动脚本。
- 更新后的 README、协作说明、贡献说明、最小 demo requirements，以及完整模型既有依赖清单与本地配置示例。
- 经过整理的运行、资源准备、项目状态、模型资源和第三方来源说明。
- 项目所有者明确特许的 `models/public/locator.json`、`models/public/motion.json` 两个去私人元信息的下游定位 readout 副本及其 registry、许可；不是原始模型或第三方权重。
- 公开合成样例生成器；视频运行时生成，不把私人样例或预置推理结果塞进发行包。

保留现有代码结构、导入、API 与输出契约；可配置 runtime、doctor 和资源准备工具用于显式资源定位，不放宽特征与来源校验。`.gitattributes` 保留既有研究源码字节，避免换行自动转换破坏来源哈希；新文档统一 LF。

## 明确排除

私有视频和标注、缓存、原始训练模型包、第三方 checkpoint 及其本地转换副本、下载源码/归档、图片/视频、日志、真实配置、临时状态、旧实验报告及未经完整许可审核的第三方源码副本。

两个 readout 发布副本是明确限定的例外，不放行原始模型、整个 `models/` 或第三方资源目录。发布副本不携带真实视频名称/指纹、私有路径、训练数据清单；`seen_in_development = null` 只表示该信息不分发，不证明输入未见。`.gitignore` 使用根目录和文档白名单；新增发布路径必须显式审核。创建文档并不等于已纳入 Git，本文档编辑阶段不自动改白名单、暂存、commit 或 push。

不安装新扫描工具，不将代码上传在线扫描服务；使用本地文本/路径/大小检查并检查实际 Git index。后续发行须检查将发布的可达历史，不能只扫描工作目录。模式扫描只能覆盖所检查的风险类型，不是任何未知秘密或隐私都绝不会出现的保证。

## 发布授权与核验

实际 owner/repository、私有或公开、目标分支、上传范围、作者身份。缺少 Git 提交身份时，仅在用户同意后配置当前仓库，不更改全局设置。仓库创建与精确分支 push 分开，完成后比较本地与远端 SHA 和实际可见性。

项目所有者已明确同意公开发布源码、采用 MIT 许可证，并使用 GitHub 账号署名与隐私提交邮箱（仅配置本仓库）。目标为 `Seele-111/jepa-system`，发布分支为 `main`。MIT 只覆盖项目所有者有权授权的自有代码及明确获准的两个下游 readout 副本，不重新授权第三方资源；I-JEPA 固定版本源码的 CC BY-NC 4.0 非商业限制仍然存在，不能宣称完整推理链路统一 MIT 或可商用。

本次没有附加 Actions、Pages、Release、协作者邀请或生产部署。公开或私有仓库都不能提交真实密钥或用户数据。

## 可复现路线与当前验收边界

- Windows 前端 + WSL 2、原生 Linux 两条路线均按资源指南使用固定模型解释器和完整 RGB profile；模型侧 Python 3.12.3、Torch 2.12.0+cu132、torchvision 0.27.0+cu132，其他依赖沿用 `requirements-models.txt`。
- 上游源码只展开固定提交 ZIP 的 `src/app` 与根层许可、说明，不展开整个训练目录；不改变推理源码树摘要，不新增训练框架。资源工具需要 `--accept-noncommercial`，大权重下载还需 `--download-checkpoints`，或传入用户自行取得的 V full、I full、R3D。启动时不自动下载。
- registry 已记录 V target_encoder / FP32、I 三分支 / BF16 的逐张量与文件 SHA 转换一致性；不能因此声称找回了所有历史训练源码或复算私有开发集成绩。历史 collection adapter 的旧 whole-file snapshot 未完整恢复，保留原 provenance，不以新摘要冒充旧版本。
- 维护者最新报告 `full-fusion-smoke-v3` 的 `optimized_true_jepa_motion` 完整退出码 0，含 RGB 与真实 JEPA、不是 fallback。新增可配置 loader 仅改 checkpoint 的 CPU mmap 暂存，避免重复 CUDA FP32 OOM；未修改原 `vjepa_predictor.py` / `optimized_jepa_extractor.py`。随后在隔离目录完成完整融合与真 JEPA 推理、新 Windows venv 的 CPU/API/浏览器流程验收；边界与可选跳过详见 [发行验收说明](REPRODUCIBLE_RELEASE.md)，不宣称所有硬件或从零重装整个 GPU 环境均已验证。

发行范围变化不改写 [项目状态](PROJECT_STATUS.md) 的具体实验记录，不替换默认实验模型，不新增生产依赖，也不授权停止现有服务、部署或外部上传。
