# 发布模型与外部资源

## 随仓库提供的 readout

| 文件 | 用途 |
| --- | --- |
| `public/locator.json` | 当前完整校正 JEPA / 运动 RF、ET / RGB TCN 融合 readout 与 decoder |
| `public/motion.json` | 现有学习式 CPU 运动 readout 与 decoder；不是规则回退 |
| `public/registry.json` | 发布文件、特征 profile、源码、外部 checkpoint 与转换的完整性声明 |
| `public/LICENSE` | 项目所有者授权的下游 readout MIT 许可，不覆盖外部 encoder |

这些是专门制作的推理发布副本，不是直接上传原始模型。树、张量、变换和解码参数保持不变；真实视频名称/内容指纹、私有报告路径、训练数据清单等不分发。

原始 `optimized_locator_v1.json`、`optimized_motion_locator_v1.json` 继续被 Git 忽略。原文件没有被本轮修改；若本地已有它们，兼容配置仍优先使用原始包。否则使用公开包。

## 校验与来源边界

- 公布文件 SHA 与推理 payload identity；改模型参数后重算 metadata 不能绕过已固定的 registry。
- 公开 RGB profile 使用逻辑资源地址。只有地址经过已审查的重定位；checkpoint SHA、预处理、特征顺序、FPS 与 runtime version 检查不放宽。
- 原 RGB profile identity 与原模型 SHA 保留为来源承诺，未把经过改写的 profile 冒充原训练文件。
- corrected extractor 的原始 SHA 保留。历史 collection adapter 的 whole-file SHA 与当前 collection 脚本不同，其旧 snapshot 尚未定位；**不能声称找回了全部历史训练源码**。原 historical SHA 不修改，当前 descriptor 实现另行固定，并与原训练-profile缓存逐帧复算证明特征行为一致。
- 发布版不携带训练成员指纹，`seen_in_development` 因此为 `null`；不能据此声称输入是未见数据。
- 数值 parity 和工作台 smoke 证明的是发行推理行为与功能，不是异常检测质量或新盲测成绩。

维护者工具 `code/export_public_models.py` 采用显式推理字段白名单，不支持任意实验模型自动发布。新导出必须完成数据权属、敏感元信息、特征来源、数值 parity 与真实推理审核；不能直接放宽 registry 或拿原始 JSON 替代审核。

## 不分发的第三方权重

V-JEPA 2.1、I-JEPA 与 torchvision R3D 仍需自行从官方来源取得，按 [资源准备指南](../docs/RESOURCE_SETUP.md) 准备。官方源码版本、checkpoint SHA 与转换规则固定在注册表；工具可重建与本地基线字节一致的 encoder-only 与 I-JEPA BF16 资源，不上传这些文件。

本项目 MIT 不改变 I-JEPA 上游源码的 CC BY-NC 限制，也不代表已为所有预训练权重取得统一商业授权。请分别检查源码、checkpoint 和数据的实际条款，见 [第三方说明](../THIRD_PARTY_NOTICES.md)。
