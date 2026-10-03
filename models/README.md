# 本地模型资源（不随源码分发）

此目录只发布说明，不包含当前训练模型或预训练 checkpoint。

## 已有本地默认文件

| 文件 | 作用 |
| --- | --- |
| `optimized_locator_v1.json` | 默认 RF / ET / RGB TCN 融合与 decoder/provenance |
| `optimized_motion_locator_v1.json` | 学习式 CPU 运动定位模型；不是规则回退，也不是真 JEPA |

这些 JSON 是真实训练模型包，包含特征 schema、权重、解码规则和开发/训练 provenance，不能当作普通示例 JSON 上传。持有原始模型时可将其恢复到本目录，`.gitignore` 会继续排除它们；不要直接编辑 provenance 来通过校验。

旧的 `.pt`/`.pkl` 研究模型也不分发。V-JEPA、I-JEPA 与 RGB R3D checkpoint 位于配置的外部模型目录，下载和使用须遵循各自上游许可。源码发布不等于取得模型或数据再分发授权。

没有模型包时，优化模式和学习式 CPU 模式不可用；仍可运行 `legacy --no-true-jepa` 的明确规则 CPU 运动回退。完整资源需求、WSL 固定路径与验收边界见 [运行指南](../docs/RUNNING.md)。
