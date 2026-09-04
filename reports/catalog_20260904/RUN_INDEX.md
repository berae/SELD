# 逐 seed 实验列表

42 个 run、84 个 run/split；均为 TAU2020，训练 folds2–6、validation fold1、evaluation 200 条。

状态为已有 checkpoint / 导出预测 / 评分，并非本次重训。完整 config、checkpoint、prediction、result 绝对路径见 [run_index.csv](run_index.csv)。

| Family | Variant | Seed | Run ID | Eval LE_CD ° | Eval LR_CD % | Eval SELD | Val SELD |
|---|---|---|---|---|---|---|---|
| EINV2 | C0.1 | 2026 | EINV2_C0_bestfix_seed2026 | 14.498 | 70.86 | 0.3692 | 0.3905 |
| EINV2 | C0.1 | 2027 | EINV2_C0_bestfix_seed2027 | 12.386 | 70.33 | 0.3619 | 0.3661 |
| EINV2 | C0.1 | 2028 | EINV2_C0_bestfix_seed2028 | 13.905 | 71.46 | 0.3375 | 0.3686 |
| EINV2 | C1 | 2026 | C1_CausalEINV2_Velocity_seed2026 | 15.128 | 69.06 | 0.3683 | 0.3914 |
| EINV2 | C1 | 2027 | C1_CausalEINV2_Velocity_seed2027 | 15.579 | 71.66 | 0.3657 | 0.3843 |
| EINV2 | C1 | 2028 | C1_CausalEINV2_Velocity_seed2028 | 12.827 | 68.64 | 0.3871 | 0.3722 |
| EINV2 | C2 | 2026 | C2_CausalEINV2_JEPA_seed2026 | 15.397 | 71.17 | 0.3624 | 0.3953 |
| EINV2 | C2 | 2027 | C2_CausalEINV2_JEPA_seed2027 | 12.791 | 68.78 | 0.3620 | 0.3684 |
| EINV2 | C2 | 2028 | C2_CausalEINV2_JEPA_seed2028 | 14.963 | 68.82 | 0.3709 | 0.3823 |
| EINV2 | C3 | 2026 | C3_CausalEINV2_Velocity_JEPA_seed2026 | 12.829 | 70.28 | 0.3547 | 0.3596 |
| EINV2 | C3 | 2027 | C3_CausalEINV2_Velocity_JEPA_seed2027 | 13.049 | 71.60 | 0.3630 | 0.3567 |
| EINV2 | C3 | 2028 | C3_CausalEINV2_Velocity_JEPA_seed2028 | 13.319 | 69.02 | 0.3668 | 0.3699 |
| EINV2 | E0 | 2026 | E0_OfflineEINV2_seed2026 | 9.849 | 78.27 | 0.2279 | 0.2555 |
| EINV2 | E0 | 2027 | E0_OfflineEINV2_seed2027 | 10.790 | 78.79 | 0.2332 | 0.2728 |
| EINV2 | E0 | 2028 | E0_OfflineEINV2_seed2028 | 10.220 | 78.46 | 0.2328 | 0.2642 |
| EINV2 | E1 | 2026 | E1_OfflineEINV2_Velocity_seed2026 | 9.794 | 77.30 | 0.2294 | 0.2550 |
| EINV2 | E1 | 2027 | E1_OfflineEINV2_Velocity_seed2027 | 10.819 | 79.16 | 0.2285 | 0.2671 |
| EINV2 | E1 | 2028 | E1_OfflineEINV2_Velocity_seed2028 | 9.837 | 79.86 | 0.2202 | 0.2612 |
| EINV2 | E2 | 2026 | E2_OfflineEINV2_JEPAConsistency_seed2026 | 11.271 | 77.40 | 0.2360 | 0.2554 |
| EINV2 | E2 | 2027 | E2_OfflineEINV2_JEPAConsistency_seed2027 | 10.708 | 77.50 | 0.2346 | 0.2665 |
| EINV2 | E2 | 2028 | E2_OfflineEINV2_JEPAConsistency_seed2028 | 13.098 | 78.79 | 0.2473 | 0.2863 |
| EINV2 | E3 | 2026 | E3_OfflineEINV2_Velocity_JEPAConsistency_seed2026 | 9.971 | 77.69 | 0.2330 | 0.2702 |
| EINV2 | E3 | 2027 | E3_OfflineEINV2_Velocity_JEPAConsistency_seed2027 | 13.103 | 76.81 | 0.2563 | 0.2876 |
| EINV2 | E3 | 2028 | E3_OfflineEINV2_Velocity_JEPAConsistency_seed2028 | 13.153 | 78.23 | 0.2534 | 0.2898 |
| Multi-ACCDOA | C0 | 2026 | paper_v1_C0_seed2026 | 15.355 | 57.34 | 0.4047 | 0.4321 |
| Multi-ACCDOA | C0 | 2027 | paper_v1_C0_seed2027 | 15.445 | 57.25 | 0.4088 | 0.4244 |
| Multi-ACCDOA | C0 | 2028 | paper_v1_C0_seed2028 | 15.233 | 55.78 | 0.4150 | 0.4251 |
| Multi-ACCDOA | C1_lambda005 | 2026 | pilot_v2_C1_lambda005_seed2026 | 15.044 | 56.68 | 0.4035 | 0.4220 |
| Multi-ACCDOA | C1_lambda005 | 2027 | promoted_v2_C1_lambda005_seed2027 | 15.424 | 54.91 | 0.4177 | 0.4185 |
| Multi-ACCDOA | C1_lambda005 | 2028 | promoted_v2_C1_lambda005_seed2028 | 15.198 | 55.83 | 0.4117 | 0.4225 |
| Multi-ACCDOA | C2_lambda005 | 2026 | pilot_v2_C2_lambda005_seed2026 | 15.098 | 56.50 | 0.4054 | 0.4203 |
| Multi-ACCDOA | C2_lambda005 | 2027 | promoted_v2_C2_lambda005_seed2027 | 15.367 | 56.76 | 0.4158 | 0.4283 |
| Multi-ACCDOA | C2_lambda005 | 2028 | promoted_v2_C2_lambda005_seed2028 | 14.816 | 55.91 | 0.4147 | 0.4217 |
| Multi-ACCDOA | C3_lambda005 | 2026 | supplemental_v2_C3_lambda005_seed2026 | 14.696 | 56.83 | 0.4018 | 0.4285 |
| Multi-ACCDOA | C3_lambda005 | 2027 | supplemental_v2_C3_lambda005_seed2027 | 14.544 | 53.87 | 0.4192 | 0.4239 |
| Multi-ACCDOA | C3_lambda005 | 2028 | supplemental_v2_C3_lambda005_seed2028 | 14.385 | 53.14 | 0.4206 | 0.4315 |
| Multi-ACCDOA | A0 | 2026 | paper_v1_A0_seed2026 | 14.802 | 63.67 | 0.3499 | 0.3679 |
| Multi-ACCDOA | A0 | 2027 | paper_v1_A0_seed2027 | 15.367 | 62.33 | 0.3636 | 0.3754 |
| Multi-ACCDOA | A0 | 2028 | paper_v1_A0_seed2028 | 14.685 | 62.21 | 0.3579 | 0.3840 |
| Multi-ACCDOA | A3_lambda005 | 2026 | supplemental_v2_A3_lambda005_seed2026 | 14.780 | 58.64 | 0.3796 | 0.3906 |
| Multi-ACCDOA | A3_lambda005 | 2027 | supplemental_v2_A3_lambda005_seed2027 | 15.309 | 61.03 | 0.3714 | 0.3767 |
| Multi-ACCDOA | A3_lambda005 | 2028 | supplemental_v2_A3_lambda005_seed2028 | 14.649 | 60.14 | 0.3687 | 0.3883 |
