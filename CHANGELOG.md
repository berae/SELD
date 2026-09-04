# Changelog

## 0.1.0 — 2026-09-04

- 从 rabbit02 导入 EINV2 独立变体及 Dynamic Multi-ACCDOA 实际代码，并保存来源路径/commit/SHA256。
- 分开 `scripts/train`、`scripts/eval`、`scripts/preprocess`；模型分目录，配置集中管理。
- Multi训练入口关闭自动test；独立测试读取run manifest/checkpoint。EINV2推理显式指定checkpoint。
- 共享DCASE官方固定版本metric；保留legacy2020语义，修复完整时长、jackknife与14类macro。
- 路径参数化、显式scalar输入、避免运行目录覆盖，保留原模型与loss。
- 导入小型历史结果和84run/split统一重评分汇总；明确15.2822→11.9407的single-seed validation范围。
- 加入入口/来源完整性、metric、模型forward/causal与Multi辅助loss测试。

未包含：数据集、权重、原始音频、完整预测；没有重训，没有重新选择历史checkpoint，也没有声称mAP或完整paper结论已验证。
