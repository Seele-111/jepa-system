# 第三方资源与未分发副本

本文件只记录外部来源和本次源码发布的边界，不为自有代码授予许可证，也不替代第三方原许可、NOTICE 或模型/数据条款。

## 本轮排除的本地源码副本

| 资源 | 官方来源 | 本地审核与分发决定 |
| --- | --- | --- |
| Depth Anything V2 / DINOv2 | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)、[DINOv2](https://github.com/facebookresearch/dinov2) | `code/depth_anything_v2/` 的部分文件有原作者版权头，但本地副本缺少完整 LICENSE 且未记录来源提交；不随本仓库分发。 |
| I-JEPA | [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | `ijepa-src/` 的源码引用根层 LICENSE，但副本未附该文件、版本来源未固定；不随本仓库分发。实际运行还依赖仓库外 I-JEPA 源码和 checkpoint。 |
| ActionFormer | [happyharrycn/actionformer_release](https://github.com/happyharrycn/actionformer_release) | `external/actionformer_release/` 是独立 Git 仓库，有其 MIT LICENSE，也包含二级移植来源及本地未提交修改；本轮整体排除，保留本地工作，不当作父仓库 submodule 上传。 |

排除不代表判断这些项目无法合法分发，而是本次没有完成其源码版本、补丁与全部继承许可的分发准备。以后要纳入，应单独固定版本、保留原版权与许可，并完成相关来源/NOTICE 核查。

历史深度分析分支仍需另行取得 Depth Anything V2 实现。某些旧提取脚本会捕获深度加载异常，程序继续运行不表示深度特征有效。本轮源码包不是所有历史算法分支的自包含环境。

## 运行时外部资源

- V-JEPA / V-JEPA2：[facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)。源码、实际 checkpoint 与预处理版本须分别记录，本仓库不分发其预训练资源。
- I-JEPA：需配置真实运行所使用的上游源码、预测器适配与 checkpoint；不能仅凭本地目录名推定实现或许可一致。
- RGB R3D 特征：依赖 [PyTorch](https://github.com/pytorch/pytorch) / [torchvision](https://github.com/pytorch/vision) 及匹配的 checkpoint/profile，不随最小前端 requirements 安装。
- 历史研究还存在 CLIP、Cosmos、VideoMAE 等可选路径；它们不是默认 demo 的完整安装说明，也未随此次源码包分发权重或数据。

自有适配器、转换器或“某模型风格”的下游头不等于分发了对应官方模型，也不等于已经核验其全部来源权属。此次静态来源审核未证明所有源文件的完整原创性。

## 许可边界

源码许可、预训练权重许可和数据许可不应互相替代。项目所有者有权授权的自有代码采用根层 [MIT License](LICENSE)。这不改变第三方源码、权重或数据的原许可，不能据此推定完整优化推理链路也获得了统一商业授权。第三方资源的使用或再分发须按实际版本和原始条款另行审核。
