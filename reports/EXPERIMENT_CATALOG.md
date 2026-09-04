# 实验结果总表

## Material Passport

- Origin Skill / Mode: experiment-agent / validate
- Origin Date: 2026-09-04
- Verification Status: ANALYZED
- Version Label: experiment_catalog_v1

## 口径与范围

当前主矩阵：**14 个变体组 × 3 seeds = 42 个 run；validation / evaluation 共 84 组预测**。本表展开已有结果，不新增训练或推理。全部是 TAU2020 FOA，train folds2–6、validation fold1（100 条）、evaluation（200 条）；seeds 2026/2027/2028。

统一主口径为 `dcase2023_micro`、完整 60 秒、1 秒 block、20° detection threshold；不是把数据集换成 DCASE2023。均值 ± sample SD（ddof=1），SD **不是**置信区间；未做显著性检验。ER / LE / SELD 越低越好，F20 / LR_CD 越高越好。F20 带空间条件，不等于纯 SED F1；纯 SED F1 和 mAP 尚无可用结果。

λ 列表示启用的每一项辅助 loss 权重：C3/E3/A3 两项分别取该值，不是二者之和。`C0.1` 为后续 bestfix baseline，在 portable configs 中映射到 EINV2 `C0`；不是历史缺失 best checkpoint 的旧 run。

## 1. 全部主矩阵结果

以下指标除末列外均来自独立 evaluation；末列是导出预测重评分的 validation SELD，不混入训练期在线 validation。

| Model / Variant | Context | λ | Seeds | LE_CD ° ↓ | LR_CD % ↑ | F20 % ↑ | ER20 ↓ | SELD ↓ | Val SELD ↓ |
|---|---|---|---|---|---|---|---|---|---|
| EINV2 C0.1 | causal | 0 | 3/3 | 13.596 ± 1.089 | 70.88 ± 0.57 | 57.54 ± 1.79 | 0.6335 ± 0.0460 | 0.3562 ± 0.0166 | 0.3751 ± 0.0134 |
| EINV2 C1 | causal | 0.2 | 3/3 | 14.511 ± 1.476 | 69.79 ± 1.63 | 55.65 ± 0.57 | 0.6685 ± 0.0407 | 0.3737 ± 0.0117 | 0.3826 ± 0.0097 |
| EINV2 C2 | causal | 0.2 | 3/3 | 14.384 ± 1.396 | 69.59 ± 1.37 | 56.21 ± 1.00 | 0.6386 ± 0.0022 | 0.3651 ± 0.0051 | 0.3820 ± 0.0134 |
| EINV2 C3 | causal | 0.2 | 3/3 | 13.066 ± 0.245 | 70.30 ± 1.29 | 57.44 ± 0.82 | 0.6508 ± 0.0188 | 0.3615 ± 0.0062 | 0.3621 ± 0.0069 |
| EINV2 E0 | noncausal | 0 | 3/3 | 10.286 ± 0.474 | 78.50 ± 0.26 | 72.30 ± 0.76 | 0.3761 ± 0.0053 | 0.2313 ± 0.0029 | 0.2641 ± 0.0087 |
| EINV2 E1 | noncausal | 0.2 | 3/3 | 10.150 ± 0.580 | 78.78 ± 1.32 | 72.96 ± 0.67 | 0.3650 ± 0.0046 | 0.2260 ± 0.0051 | 0.2611 ± 0.0060 |
| EINV2 E2 | noncausal | 0.2 | 3/3 | 11.692 ± 1.249 | 77.90 ± 0.77 | 70.96 ± 1.34 | 0.3809 ± 0.0153 | 0.2393 ± 0.0070 | 0.2694 ± 0.0157 |
| EINV2 E3 | noncausal | 0.2 | 3/3 | 12.076 ± 1.823 | 77.58 ± 0.72 | 69.94 ± 1.96 | 0.3984 ± 0.0197 | 0.2476 ± 0.0127 | 0.2825 ± 0.0107 |
| Multi-ACCDOA C0 | causal | 0 | 3/3 | 15.344 ± 0.106 | 56.79 ± 0.88 | 48.98 ± 0.69 | 0.6105 ± 0.0064 | 0.4095 ± 0.0052 | 0.4272 ± 0.0043 |
| Multi-ACCDOA C1_lambda005 | causal | 0.05 | 3/3 | 15.222 ± 0.191 | 55.81 ± 0.89 | 48.88 ± 1.02 | 0.6062 ± 0.0086 | 0.4110 ± 0.0071 | 0.4210 ± 0.0022 |
| Multi-ACCDOA C2_lambda005 | causal | 0.05 | 3/3 | 15.094 ± 0.275 | 56.39 ± 0.43 | 48.79 ± 0.85 | 0.6157 ± 0.0140 | 0.4119 ± 0.0057 | 0.4234 ± 0.0043 |
| Multi-ACCDOA C3_lambda005 | causal | 0.05 | 3/3 | 14.542 ± 0.155 | 54.61 ± 1.95 | 49.04 ± 1.34 | 0.6112 ± 0.0102 | 0.4139 ± 0.0105 | 0.4279 ± 0.0038 |
| Multi-ACCDOA A0 | noncausal | 0 | 3/3 | 14.951 ± 0.365 | 62.74 ± 0.81 | 54.80 ± 0.87 | 0.5207 ± 0.0108 | 0.3571 ± 0.0069 | 0.3758 ± 0.0081 |
| Multi-ACCDOA A3_lambda005 | noncausal | 0.05 | 3/3 | 14.913 ± 0.349 | 59.94 ± 1.21 | 53.06 ± 0.66 | 0.5401 ± 0.0075 | 0.3733 ± 0.0057 | 0.3852 ± 0.0074 |

## 2. 与同 seed baseline 的配对差

Δ = variant − baseline；比例差用百分点（pp）。这些是描述性配对差，不是显著性结论，也不把三个 seed 当三个独立数据集。

| Variant | Baseline | ΔLE ° | ΔLR pp | ΔF20 pp | ΔSELD | LE↓ seeds | SELD↓ seeds |
|---|---|---|---|---|---|---|---|
| EINV2 C1 | C0.1 | 0.915 ± 2.150 | -1.10 ± 2.16 | -1.89 ± 2.14 | 0.0175 ± 0.0279 | 1/3 | 1/3 |
| EINV2 C2 | C0.1 | 0.788 ± 0.340 | -1.30 ± 1.49 | -1.33 ± 2.43 | 0.0089 ± 0.0215 | 0/3 | 1/3 |
| EINV2 C3 | C0.1 | -0.530 ± 1.167 | -0.58 ± 1.85 | -0.10 ± 2.61 | 0.0053 ± 0.0222 | 2/3 | 1/3 |
| EINV2 E1 | E0 | -0.136 ± 0.218 | 0.27 ± 1.19 | 0.66 ± 0.99 | -0.0053 ± 0.0070 | 2/3 | 2/3 |
| EINV2 E2 | E0 | 1.406 ± 1.480 | -0.61 ± 0.84 | -1.33 ± 1.50 | 0.0080 ± 0.0066 | 1/3 | 0/3 |
| EINV2 E3 | E0 | 1.789 ± 1.477 | -0.93 ± 0.93 | -2.36 ± 1.32 | 0.0163 ± 0.0097 | 0/3 | 0/3 |
| Multi-ACCDOA C1_lambda005 | C0 | -0.123 ± 0.164 | -0.98 ± 1.23 | -0.10 ± 0.74 | 0.0014 ± 0.0065 | 3/3 | 2/3 |
| Multi-ACCDOA C2_lambda005 | C0 | -0.251 ± 0.169 | -0.40 ± 0.49 | -0.19 ± 0.70 | 0.0024 ± 0.0039 | 3/3 | 1/3 |
| Multi-ACCDOA C3_lambda005 | C0 | -0.803 ± 0.126 | -2.18 ± 1.49 | 0.06 ± 0.80 | 0.0044 ± 0.0067 | 3/3 | 1/3 |
| Multi-ACCDOA A3_lambda005 | A0 | -0.038 ± 0.018 | -2.80 ± 1.97 | -1.74 ± 1.24 | 0.0161 ± 0.0119 | 3/3 | 0/3 |

## 3. 静态 / 动态定位诊断

**下表不是官方 LE_CD / LR_CD**：按 `(file, frame, class)` 做 Hungarian 角度匹配。frame LE 仅统计匹配成功项；matched recall 的分母是该组全部 GT source-frames；Recall@20 额外要求角误差不超过 20°。三 seed 均值 ± SD。

motion group 来自同 `(class, source)` 的连续 GT segment：若 segment 中存在相邻帧角位移 > 1e−6°，该 segment 的 source-frames 标记 dynamic，否则有至少两帧时为 static；不是逐瞬间速度阈值分组。Evaluation：static 64,036、dynamic 53,184，占 45.37%；不支持笼统称“动态样本很少”。Validation 对应 31,428 / 26,274。不同 class 分布可能不同，不据汇总推每类都改善。

| Variant | Group | GT frames | frame LE ° ↓ | matched recall % ↑ | Recall@20 % ↑ |
|---|---|---|---|---|---|
| EINV2 C0.1 | static | 64036 | 11.612 ± 0.675 | 58.80 ± 0.79 | 53.89 ± 0.75 |
| EINV2 C0.1 | dynamic | 53184 | 13.712 ± 1.328 | 61.92 ± 0.54 | 54.37 ± 0.89 |
| EINV2 C1 | static | 64036 | 12.622 ± 1.230 | 58.66 ± 1.20 | 53.28 ± 0.64 |
| EINV2 C1 | dynamic | 53184 | 14.484 ± 1.716 | 60.50 ± 1.54 | 52.81 ± 1.27 |
| EINV2 C2 | static | 64036 | 12.713 ± 1.548 | 58.17 ± 1.00 | 52.28 ± 1.45 |
| EINV2 C2 | dynamic | 53184 | 14.279 ± 1.527 | 59.83 ± 1.21 | 52.09 ± 1.49 |
| EINV2 C3 | static | 64036 | 11.274 ± 0.025 | 58.97 ± 1.57 | 54.60 ± 1.88 |
| EINV2 C3 | dynamic | 53184 | 13.034 ± 0.150 | 60.34 ± 0.80 | 53.79 ± 0.88 |
| EINV2 E0 | static | 64036 | 8.611 ± 0.328 | 71.43 ± 0.20 | 67.86 ± 0.44 |
| EINV2 E0 | dynamic | 53184 | 10.918 ± 0.663 | 73.03 ± 0.66 | 66.48 ± 0.52 |
| EINV2 E1 | static | 64036 | 8.537 ± 0.416 | 72.12 ± 2.05 | 68.38 ± 2.10 |
| EINV2 E1 | dynamic | 53184 | 10.583 ± 0.714 | 73.11 ± 0.99 | 67.00 ± 0.93 |
| EINV2 E2 | static | 64036 | 9.761 ± 1.361 | 70.95 ± 0.39 | 66.46 ± 0.91 |
| EINV2 E2 | dynamic | 53184 | 12.141 ± 1.140 | 72.46 ± 1.29 | 65.05 ± 0.47 |
| EINV2 E3 | static | 64036 | 10.164 ± 1.575 | 70.43 ± 0.64 | 65.75 ± 1.73 |
| EINV2 E3 | dynamic | 53184 | 12.360 ± 1.719 | 71.43 ± 0.46 | 63.99 ± 1.99 |
| Multi-ACCDOA C0 | static | 64036 | 12.823 ± 0.606 | 46.18 ± 0.77 | 40.58 ± 0.93 |
| Multi-ACCDOA C0 | dynamic | 53184 | 16.427 ± 0.401 | 51.68 ± 1.85 | 38.91 ± 1.15 |
| Multi-ACCDOA C1_lambda005 | static | 64036 | 13.108 ± 0.542 | 44.78 ± 0.80 | 38.89 ± 1.02 |
| Multi-ACCDOA C1_lambda005 | dynamic | 53184 | 16.428 ± 0.165 | 51.63 ± 1.59 | 39.20 ± 1.70 |
| Multi-ACCDOA C2_lambda005 | static | 64036 | 12.718 ± 0.635 | 46.02 ± 0.13 | 40.33 ± 0.94 |
| Multi-ACCDOA C2_lambda005 | dynamic | 53184 | 16.335 ± 0.297 | 52.44 ± 0.82 | 39.83 ± 0.64 |
| Multi-ACCDOA C3_lambda005 | static | 64036 | 12.267 ± 0.416 | 44.20 ± 2.52 | 39.22 ± 2.45 |
| Multi-ACCDOA C3_lambda005 | dynamic | 53184 | 15.372 ± 0.552 | 50.56 ± 2.36 | 39.73 ± 1.44 |
| Multi-ACCDOA A0 | static | 64036 | 12.400 ± 0.321 | 57.64 ± 0.59 | 50.34 ± 0.74 |
| Multi-ACCDOA A0 | dynamic | 53184 | 16.802 ± 0.441 | 62.72 ± 1.37 | 46.77 ± 0.64 |
| Multi-ACCDOA A3_lambda005 | static | 64036 | 12.742 ± 0.362 | 55.17 ± 1.52 | 48.10 ± 1.11 |
| Multi-ACCDOA A3_lambda005 | dynamic | 53184 | 16.281 ± 0.383 | 61.42 ± 0.38 | 46.39 ± 0.30 |

## 4. 完成状态、缺口与历史边界

| 范围 | 已有证据 / 状态 |
|---|---|
| EINV2 C0/C1/C2/C3、E0/E1/E2/E3 | 每个 3 seeds，validation + evaluation 已有评分 |
| Multi C0/C1/C2/C3、A0/A3 | 每个 3 seeds；当前非零辅助权重 .05，validation + evaluation 已有评分 |
| Multi A1 / A2 | 有实现和 portable config；未进入本次统一结果矩阵，不能标记为已有三 seed 结果 |
| 历史 λ=.2、smoke/pilot | 不混入 λ=.05 主矩阵；本表不是所有历史试跑的穷尽目录 |
| 历史 EINV2 15.28° → 11.94° | single-seed/fold1、旧 metric；[历史证据](HISTORICAL_LE.md)独立保留，不替代本表 |
| E0 seed2026 | 归档 checkpoint 与预测可核查；最初启动/退出日志缺失，训练来源仅 PARTIALLY VERIFIED |
| 本批补齐 | 历史执行凭据：12 次补训练（8 EINV2、4 Multi）、77 项评测、2 项 motion 作业；不把 91 作业当 91 独立实验。RB05 本批准备队列未启动，不重复计数 |
| 模型版本与评分版本 | 历史 checkpoint 按旧 validation 规则选择，再统一重评分；不能声称已按新 metric 重训/重选。当前 portable config 是复现入口，不是原始 run config |

## 5. 核验结论与限制

当前可 **Share with caveats**：已从原始 run 行独立重算全部主口径均值、sample SD、配对差和 SELD 公式；本轮服务器只读核查 84 行 config/checkpoint/result 均存在，预测文件集合全部匹配参考。只检查 checkpoint 文件存在及大小，未反序列化或重新训练。配对诊断已确认使用同一预测目录。

模型/方法相关 62 个源文件中，54 个与仓库逐字节一致；8 个差异为已登记的 scalar 路径参数化（7）与 metric protocol 标识（1），不是架构/loss 改动。文件证据成立不等于独立复现或统计显著。

11/11 fallacy checks：

| 检查 | 本表处理 / 剩余限制 |
|---|---|
| Simpson's paradox | 全局与 motion 分层分开；未声称无 class-level 反转 |
| Ecological fallacy | run 均值不推断每个声源收益 |
| Berkson selection | LE 条件于匹配，须与 recall 同看 |
| Collider bias | 不对匹配子集作无偏总体定位因果解释 |
| Base rate neglect | 明列 static/dynamic GT 分母 |
| Regression to mean | 保留全部 3 seeds，不以最优 seed 宣称泛化 |
| Survivorship | A1/A2 缺口、E0 日志缺失、旧 best 缺失显式标记 |
| Look-elsewhere | 所有 14 主组展开；不报告筛选后“显著”结论 |
| Forking paths | 保留旧权重选择、不同 λ / metric / split 的界限 |
| Correlation / causation | 不据辅助 loss 比较排除架构、初始化、mask 等混杂 |
| Reverse causality | 非横断面因果研究，本项不适用 |

## Evidence Index

- [42 个逐 seed run](catalog_20260904/RUN_INDEX.md)；[84 行完整 config/checkpoint/result 路径](catalog_20260904/run_index.csv)。
- [原始 336 行评分](aligned_20260904/aligned_run_metrics.csv)、[112 组多协议汇总](aligned_20260904/aligned_aggregate_metrics.csv)、[原配对差](aligned_20260904/aligned_paired_seed_deltas.csv)：除主 micro 外还含 macro / 两种 legacy profile，不能横向混比。
- [本轮服务器证据检查](catalog_20260904/source_audit.json)、[本地重算检查](catalog_20260904/catalog_checks.json)。
- [168 行 motion 诊断](catalog_20260904/completion_motion_per_run.csv)、[分母](catalog_20260904/completion_motion_support.csv)。
- [12 次新增训练记录](catalog_20260904/completion_new_training.csv)、[历史作业 QA](catalog_20260904/completion_qa_checks.json)：仅作执行来源，不把其中旧 metric 列改名为新结果。
- [可复算生成脚本](../scripts/reports/build_experiment_catalog.py)、[只读源文件检查脚本](../scripts/reports/audit_result_sources.py)。
- [方法与架构图](../docs/METHOD_AND_ARCHITECTURE.md)。

复算：`python scripts/reports/build_experiment_catalog.py`。仅读取仓库快照并更新派生表，不访问 GPU、数据集或实验目录。
