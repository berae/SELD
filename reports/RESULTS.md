# 统一 metric 结果摘要

下面均为 TAU2020 **evaluation、seeds2026/2027/2028的均值**，`dcase2023_micro`、完整60秒。单run与standard deviation以随附CSV为准；没有在本版重训或重新选择checkpoint。

| Family | Variant | LE_CD ↓ | LR_CD ↑ | F20 ↑ | SELD ↓ |
|---|---|---:|---:|---:|---:|
| EINV2 causal | C0.1 bestfix | 13.5961 | 0.708838 | 0.575380 | 0.356200 |
| EINV2 causal | C3 | 13.0656 | 0.702999 | 0.574414 | 0.361483 |
| EINV2 offline | E0 | 10.2864 | 0.785049 | 0.722957 | 0.231309 |
| EINV2 offline | E1 | 10.1502 | 0.787751 | 0.729597 | 0.225998 |
| Multi-ACCDOA causal | C0 | 15.3443 | 0.567902 | 0.489796 | 0.409519 |
| Multi-ACCDOA causal | C3 lambda005 | 14.5418 | 0.546126 | 0.490406 | 0.413870 |

这两种架构的C3均表现为全局LE下降、LR下降；不能把LE改善等同于所有指标成功。F20带空间条件，不是纯SED F1；当前不据此声称SED F1或mAP已保持。

其他实际完成变体（C1/C2/E2/E3/A0/A3等）见 `aligned_20260904/aligned_aggregate_metrics.csv`。该文件含不同split与4个protocol，比较时必须同时筛选相同family、split、protocol、weight。

证据：

- `aligned_20260904/aligned_run_metrics.csv`：84 run/split × 4 profiles。
- `aligned_20260904/aligned_aggregate_metrics.csv`：分组统计。
- `aligned_20260904/aligned_paired_seed_deltas.csv`：相同seed配对差。
- `aligned_20260904/evidence_index.csv`：历史config/checkpoint/prediction/source路径。
- `aligned_20260904/provenance_audit.json`：旧结果重现审计。
- `aligned_20260904/qa_summary.json`、`integration_tests.json`：此前统一重评分的验证，非本次目录整理测试。

源代码状态与旧结果时间点不同：2026-09-04切换新validation metric，此前权重仍按旧metric选择。历史预测重新评分验证了metric对齐，不证明重新训练会得到相同权重。
