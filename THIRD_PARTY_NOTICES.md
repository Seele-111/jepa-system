# 第三方资源与未分发副本

本文件记录外部来源和本次发行的边界，不替代第三方原许可、NOTICE 或模型/数据条款。项目所有者有权授权的自有代码采用根层 [MIT License](LICENSE)；这不表示完整模型链路统一获得 MIT 或商业授权。

## 两个获授权的发布模型例外

所有者已明确允许分发 `models/public/locator.json` 与 `models/public/motion.json` 两个去私人元信息的下游定位 readout 副本及其 registry、许可。学习参数、特征顺序和 decoder 保持不变；真实视频名称/内容指纹、私有路径和训练数据清单不分发。它们不是第三方预训练 encoder，也不是直接上传原始训练模型。

该例外只覆盖这两个经过审核的发布副本。原始模型 JSON、第三方源码副本/权重、数据、标注、缓存、日志和带私人来源的旧报告仍不属于上传范围。发布副本的 MIT 许可见 [models/public/LICENSE](models/public/LICENSE)，不覆盖外部 encoder、checkpoint 或数据。

## 当前运行资源的官方来源

版本、源码树摘要、checkpoint SHA 与转换声明以 [registry](models/public/registry.json) 为准；准确获取命令见 [资源准备指南](docs/RESOURCE_SETUP.md)。

| 资源 | 固定官方来源与许可 | 本仓库分发决定 |
| --- | --- | --- |
| V-JEPA 2.1 | [facebookresearch/vjepa2 @ 204698b45b3712590f06245fbfba32d3be539812](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812)；[LICENSE](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/LICENSE) 为 MIT，固定 [README 许可说明](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/README.md#license) 还列出若干源码的 Apache-2.0 等继承声明，须保留 | 不分发上游源码/权重；用户自行获取固定源码、ViT-g 384 full，并在本地重建 encoder-only。权重的实际使用/再分发条款另行审阅。 |
| I-JEPA | [facebookresearch/ijepa @ 52c1ae95d05f743e000e8f10a1f3a79b10cff048](https://github.com/facebookresearch/ijepa/tree/52c1ae95d05f743e000e8f10a1f3a79b10cff048)；[LICENSE](https://github.com/facebookresearch/ijepa/blob/52c1ae95d05f743e000e8f10a1f3a79b10cff048/LICENSE) 为 **CC BY-NC 4.0（非商业）** | 不分发上游源码/权重；用户获取固定官方 `IN22K-vit.g.16-600e.pth.tar`，按固定三分支 BF16 转换。源码、checkpoint 和数据条款不能互相替代。 |
| RGB R3D-18 | [torchvision 0.27.0 模型定义](https://github.com/pytorch/vision/blob/v0.27.0/torchvision/models/video/resnet.py) 的 `R3D_18_Weights.KINETICS400_V1`；[源码 LICENSE](https://github.com/pytorch/vision/blob/v0.27.0/LICENSE) 为 BSD-3-Clause | 不分发 `r3d_18-b3b3357e.pth`；按 registry 的官方 PyTorch URL 获取并核对 SHA。源码许可不自动授权预训练权重及相关数据。 |

I-JEPA 固定 README 将该 ImageNet-22K / ViT-g/16 checkpoint 表项写为 44 epochs，链接文件名却含 `600e`；使用固定链接和 SHA，不据文件名推断训练轮次。V、I 的本地转换文件也属于不分发的第三方衍生产物，而非上述 readout 例外。

资源准备工具须显式使用 `--accept-noncommercial`；该标志不是商业授权。下载大 checkpoint 还须显式 `--download-checkpoints`，或提供用户自行取得的 V full、I full 与 R3D 三份输入。工具只从固定源码 ZIP 展开 `src/app` 和根层许可、说明文件，不展开整个训练目录，不安装上游训练依赖；推理源码树摘要仍须与 registry 相同。启动前端、doctor 或 Worker 不自动下载资源。

## 本轮排除的历史本地源码副本

| 资源 | 官方来源 | 本地审核与分发决定 |
| --- | --- | --- |
| Depth Anything V2 / DINOv2 | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)、[DINOv2](https://github.com/facebookresearch/dinov2) | `code/depth_anything_v2/` 的部分文件有原作者版权头，但历史本地副本缺少完整 LICENSE 且未记录来源提交；不随本仓库分发。 |
| 历史 I-JEPA 本地副本 | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | 历史 `ijepa-src/` 未附其引用的根层 LICENSE、来源提交未固定；继续排除。它不是上述工具取得、保留 LICENSE 且验证固定摘要的运行资源，不能拿旧目录名冒充通过核验。 |
| ActionFormer | [happyharrycn/actionformer_release](https://github.com/happyharrycn/actionformer_release) | `external/actionformer_release/` 是独立 Git 仓库，有其 MIT LICENSE，也包含二级移植来源及本地未提交修改；本轮整体排除，保留本地工作，不当作父仓库 submodule 上传。 |

排除不代表判断这些项目无法合法分发，而是本次没有完成其源码版本、补丁与全部继承许可的分发准备。以后要纳入，应单独固定版本、保留原版权与许可，并完成相关来源/NOTICE 核查。

历史深度分析分支仍需另行取得 Depth Anything V2 实现。某些旧提取脚本会捕获深度加载异常，程序继续运行不表示深度特征有效。历史 CLIP、Cosmos、VideoMAE 等可选路径也不是当前默认 demo 的安装范围，未随本仓库分发其权重或数据。

## 许可边界

源码许可、预训练权重许可和数据许可应分别审阅。自有适配器、转换器或“某模型风格”的下游头不等于官方模型，也不改变第三方条款。此次来源核对不证明所有历史源文件的完整原创性，不提供全链商用保证，也不授权未经审核的资源再分发。尤其不能用项目 MIT 掩盖 I-JEPA 源码的非商业限制。
