# Changelog

## 0.3.0 — 2026-09-07

- 汇总68个统一主run、131个已评分split，并收录旧Multi λ=.2/λ=.02、STARSS与失败记录；保留原始metric口径。
- 合并服务器已使用的low-JEPA/dynamicmask配置能力，不修改模型/loss数学实现。
- 新增显式recipe/host路径、新D1/D3队列别名、三个runtime的SHA256精确恢复工具。
- 新归档168份历史项目源文件/配置；旧训练目录、checkpoint与数据保持不变。
- 更正：Multi A1/A2旧λ=.2已有三seed；缺的是λ=.05完整结果。90epoch新版run不是早停。

## 2026-09-05 — `einv2_rb05_dynamicmask_v1`

- Added an opt-in `velocity_min_norm` filter for the EINV2 velocity auxiliary loss.
- Kept the default at `0.0`, which preserves the audited all-valid-pairs objective.
- Added a two-run pilot: dynamic-pair velocity-only and dynamic-pair velocity + JEPA `0.05`.
- Inference, DCASE-aligned scoring, model architecture, data, and checkpoint selection are unchanged.

## 0.2.0 — 2026-09-04

- 用户确认当前100 ms帧口径，保留原75 ms输入依赖；不再宣称相对帧起点零前视。
- 新增隔离EINV2 causal runtime与独立train/eval入口；历史模型、checkpoint和结果不覆盖。
- 统一归一化；新scaler仅用folds2–6并保存完整500文件清单；C0–C3共享初始化和dropout路径。
- teacher独立tPIT匹配；保留主要损失、权重、horizons。官方validation选best，StepLR在完整epoch边界执行。
- 新增7项回归测试与有限12-run队列；smoke通过不代表正式实验完成。

## Unreleased — 2026-09-04

- 展开 14 组实验主矩阵与 42 个逐 seed run；保存 84 行证据索引、静态/动态诊断及独立重算检查。
- 新增 2 张方法示意图、4 张 causal/noncausal 架构图及 SVG 生成源码，覆盖 16 个 portable recipes，按实际 mask、head、normalization 和 EMA 实现绘制。

- 新增 rabbit02 / RB05 数据资产表、六套原始数据与派生缓存的只读快照、标签覆盖和 TAU2020 迁移核对结果。
- 新增独立盘点脚本、复核 notebook 和 5 项测试；不修改模型、训练配置、metric 或服务器原始数据，代码版本仍为 0.1.0。

## 0.1.0 — 2026-09-04

- 从 rabbit02 导入 EINV2 独立变体及 Dynamic Multi-ACCDOA 实际代码，并保存来源路径/commit/SHA256。
- 分开 `scripts/train`、`scripts/eval`、`scripts/preprocess`；模型分目录，配置集中管理。
- Multi训练入口关闭自动test；独立测试读取run manifest/checkpoint。EINV2推理显式指定checkpoint。
- 共享DCASE官方固定版本metric；保留legacy2020语义，修复完整时长、jackknife与14类macro。
- 路径参数化、显式scalar输入、避免运行目录覆盖，保留原模型与loss。
- 导入小型历史结果和84run/split统一重评分汇总；明确15.2822→11.9407的single-seed validation范围。
- 加入入口/来源完整性、metric、模型forward/causal与Multi辅助loss测试。

未包含：数据集、权重、原始音频、完整预测；没有重训，没有重新选择历史checkpoint，也没有声称mAP或完整paper结论已验证。
