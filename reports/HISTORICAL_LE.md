# 历史 causal EINV2：15.28 → 11.94

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-04
- Verification Status: ANALYZED
- Version Label: historical_le_v1

原始日志确实有这组结果，精确值不是11.93而是 **11.9407221446**。

| Run | Seed / Split | Epoch | ER20 | F20 | LE20 (°) | 旧LR20（实际LF） | SELD20 |
|---|---|---:|---:|---:|---:|---:|---:|
| C0p1_CausalEINV2_fix_seed2026 | 2026 / fold1 validation | 81 | 0.627111 | 0.549211 | 15.282208 | 0.659733 | 0.375767 |
| C3_CausalEINV2_Velocity_JEPA_seed2026 | 2026 / fold1 validation | 77 | 0.566560 | 0.601258 | 11.940722 | 0.671859 | 0.339945 |

描述性差值 **−3.341486°（约−21.87%）**。它不是三 seed evaluation均值，不能推成普遍改善幅度，也没有做显著性检验。

## 直接证据

- 本仓库 `reports/historical/C0_seed2026_validation_metrics.csv`：第82行（含header），epoch81。
- 本仓库 `reports/historical/C3_seed2026_validation_metrics.csv`：第78行，epoch77。
- 原始路径：`rabbit02:/work/zhanghc/Myllm/SELD/EINV2/C0p1_CausalEINV2_fix_seed2026/out_train/ein_seld/C0p1_CausalEINV2_fix_seed2026/checkpoints/metrics_statistics.csv`。
- 原始路径：`rabbit02:/work/zhanghc/Myllm/SELD/EINV2/C3_CausalEINV2_Velocity_JEPA_seed2026/out_train/ein_seld/C3_CausalEINV2_Velocity_JEPA_seed2026/checkpoints/metrics_statistics.csv`。
- 配置：`configs/einv2/historical/C0/c0p1_causal_ivfix_smallhead_seed2026.yaml`、`configs/einv2/historical/C3/c3_causal_velocity_jepa_seed2026.yaml`。

C0.1 epoch81 checkpoint **未保留**；旧best/latest保存问题导致留存权重对应epoch90。C3 epoch77权重保留。因此日志事实可确认，但无法用缺失的baseline epoch81权重做统一metric重评分；历史比较为PARTIALLY REPRODUCIBLE。

后续 C0 bestfix 三 seed 与这条旧C0.1日志不是同一个run。当前共同metric的evaluation均值为EINV2 C0.1 **13.5961°**、C3 **13.0656°**，详见 `RESULTS.md`。

## 解释限制（11/11 fallacy checks）

| 检查 | 结果 |
|---|---|
| Simpson's paradox | 此两行无分层，不能推静态/动态各组方向 |
| Ecological fallacy | 不从run平均推每个source都改善 |
| Berkson selection | 属validation pilot，不代表未选测试总体 |
| Collider bias | 此描述性比较未控制共同结果变量 |
| Base rate neglect | 不用总LE推动态数据占比或LR变化 |
| Regression to mean | single seed、选epoch，泛化幅度未确定 |
| Survivorship | baseline最佳epoch权重缺失已显式披露 |
| Look-elsewhere | 两个selected epochs，不作显著性结论 |
| Forking paths | 旧metric、checkpoint选择与后续bestfix区别已标注 |
| Correlation/causation | 不据这两行排除所有训练混杂 |
| Reverse causality | 非横断面因果推断，本项不适用 |
