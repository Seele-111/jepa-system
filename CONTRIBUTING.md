# 贡献与维护

本项目是研究原型，当前发行包含源码、公开合成样例生成器，以及项目所有者特许的两个去私人元信息的下游定位模型发布副本，不是完整训练数据/第三方权重包。项目所有者有权授权的自有代码及获授权 readout 采用 [MIT License](LICENSE)；第三方资源各自遵循原许可，尤其 I-JEPA 源码的 CC BY-NC 限制不被 MIT 覆盖。贡献自有代码请遵循项目许可，并确认有权提交对应内容。

## 开始修改前

1. 阅读 `README.md`、`AGENTS.md`、[运行指南](docs/RUNNING.md)、[资源准备指南](docs/RESOURCE_SETUP.md) 和 [项目状态](docs/PROJECT_STATUS.md)，确认本次修改的是演示、portable 推理还是研究实验；真实模型还应阅读 [模型说明](models/README.md) 与 [第三方说明](THIRD_PARTY_NOTICES.md)。
2. 保持现有目录与导入方式，避免无关重构或格式化。保留 API、输出契约、FPS/区间约定、特征顺序、模型签名及来源校验。
3. 不新增生产依赖，不调整默认模型、固定分组/OOF 边界、正常误报与空候选护栏，除非有明确授权与证据。
4. 本地视频、标注、特征缓存、原始训练模型、checkpoint 和旧报告不属于普通源码提交范围。两个发布 readout 是明确限定的例外，不扩大到其他训练包或第三方资源。

## 检查

从项目根目录，在已安装相应依赖的解释器中执行与修改风险相关的检查：

```text
python -B code/model_doctor.py
python -m unittest discover -s code -p "test_demo.py" -v
python -m unittest discover -s code -p "test_optimized_locator.py" -v
```

修改发行包校验、配置或公开样例时，再选择相应的聚焦检查：

```text
python -m unittest discover -s code -p "test_public_release.py" -v
python -m unittest discover -s code -p "test_public_samples.py" -v
```

默认 doctor 只核查模型包；上述检查不下载权重，不代表 GPU 推理或独立泛化验收通过。缺少可选 sklearn 时，应记录跳过项；不要为普通文档小改动要求全库训练环境。完整模型修改需先按资源指南配置固定的 Python 3.12.3 / Torch 2.12.0+cu132 / torchvision 0.27.0+cu132 与完整 RGB profile，再显式运行：

```text
python -B code/model_doctor.py --model-runtime
```

该检查使用配置的 native/WSL 模型解释器，仅检查源码、checkpoint、版本与 CUDA 可用性；仍不能替代真实新视频的完整推理。分别记录 full fusion、legacy、干净环境、数值 parity 和 UI smoke 的结果；其中一项成功不能扩写为其余项通过。现有 full-fusion smoke 证据和未完成项见项目状态的发行说明。

## 发布模型与资源变更

- 允许分发的模型仅为 `models/public/locator.json`、`models/public/motion.json` 两个审核后的发布副本及其 registry、许可。使用 `code/export_public_models.py` 的明确推理字段白名单，不直接上传原始 JSON，也不把任意实验模型套上 publication metadata。
- 发布前审核数据权属与私人元信息；核对模型文件 SHA、推理 payload identity、feature profile、源码与 checkpoint 声明、转换逐张量/文件一致性，以及真实推理。参数或代码变化不能靠重算 metadata 绕过固定 registry。
- 仅允许已审查的资源地址重定位；不放宽 checkpoint SHA、预处理、特征顺序、FPS 或 runtime version。历史 collection adapter 的旧 whole-file snapshot 未完整恢复，不能把当前实现摘要冒充历史源码；见模型说明。
- 发布版不携带开发视频指纹，`seen_in_development` 为 `null`，不能因此声称输入一定未见过。合成样例没有真实评测标注，不构成检测质量证据。
- 新资源依然由用户从官方来源自行取得。`--accept-noncommercial` 是 I-JEPA 非商业限制的确认，不是商业许可；`--download-checkpoints` 明确允许大下载。不让导入或启动自动获取权重，不新增训练框架或生产依赖。
- `.gitignore` 的根目录/文档白名单不是授权证明。新增文档或发布资源路径须单独审核纳入；不借模型例外放行整个资源目录，也不在贡献中自动 push、部署或重写历史。

## 算法实验

- 先根据基线错误提出可证伪的假设，再固定分组、对照、选择规则和升级标准。
- 使用内容分组与严格折外证据，不能把用于选参数的数据再当独立盲测。
- 同时记录定位质量、正常误报、异常无候选和实际推理兼容性。
- 不为了分数改变原标注，不删失败案例，不用诊断 AUC 替代产品成绩。
- 新产物写入独立目录，保留已有默认模型与冻结证据。发行文档变动不能改写具体实验记录。

## 提交与反馈

提交应说明修改目的、范围、检查结果和已知限制。不要把私有数据、完整请求体、Token、邮箱或原始日志放进提交/问题反馈；只提供必要错误码和脱敏复现。发布前检查整个暂存区及将发布的可达历史，而不是依赖 `.gitignore` 自动证明安全。不要为贡献检查停止或重启他人正在运行的 demo/Worker。
