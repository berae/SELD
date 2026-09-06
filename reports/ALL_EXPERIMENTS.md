# SELD 全部实验结果与代码整理（2026-09-07）

证据来自 rabbit02 `/work/zhanghc/Myllm/SELD` 与 RB05 `/home/zhanghc/SELD` 的实际文件；不采用聊天中的成绩作为数据源。

[Dashboard](summary_20260907/DASHBOARD.md) · [代码地图](../docs/CODE_MAP.md)

## 1. 总结

主结果矩阵 **68 个 run**（历史42 + 重审/补充26），**131 个已评分 run×split**；新版26个正式run均完成90 epochs，21个完成独立evaluation，5个仅validation。另列 7 个新版smoke/初始化记录和 175 条旧版manifest/训练曲线/评估记录；后者包含重复评测和迁移副本，不能相加当独立实验数。

VERIFIED：新版 velocity-only 在两台机器上都呈现平均 LE 改善，但 LR 下降。低权重 JEPA 联合方案比高权重方案 validation 更好；尚未证明优于 velocity-only 或稳定互补。不能概括为所有 EINV2 成功，亦不能说 Multi-ACCDOA 完全无定位收益。

## 2. 研究目标与实际方法

目标：以显式 DOA 一阶差分 velocity 监督与隐空间未来预测互补，提高定位 LE，尽量维持 LR/F20。EINV2 与 Multi-ACCDOA 均有 velocity head、latent projector/predictor、EMA teacher 和辅助 loss；不把已有 DOA derivative 思路当全新贡献。独立 acceleration 监督并非本次主矩阵方法。

新版 EINV2 采用同一 C3 架构做 C0/C1/C2/C3 配对（零权重关闭梯度）；PIT 对齐后计算 velocity 与 teacher/online 轨道 latent。JEPA 是训练期增添的结构，不只是标量 loss；当前推理仍会计算部分辅助输出，不能声称已裁剪到零推理开销。

## 3. 统一评价口径与设置

主表采用固定 DCASE2023 official core + micro；这是 metric 版本，不是 DCASE2023 数据。F20 为 location-dependent F，不是纯 SED F1；未收集 mAP。旧 LR20 可能是 localization F，原始表保留原名，不冒充 LR_CD。

TAU2020 FOA；train folds2–6、validation fold1、evaluation独立200条录音；seeds 2026/2027/2028。新版 EINV2：24 kHz、FFT1024/hop600、256 mel + intensity，4 s独立chunk，100 ms label内允许75 ms依赖；不是零延迟/跨chunk流式缓存。train-only scaler、FOA reorder=False、batch32、Adam amsgrad lr5e-4、epoch80后×0.1、固定90 epochs，按 validation SELD_LR 最小选best，**不是90前早停**。

velocity λ=.2，JEPA λ=.2或.05；预测 horizons=[1,3,5] frames，latent128、predictor hidden256、EMA=.996。D1/D3仅对 target velocity norm>1e-6 的有效pair施加velocity loss。完整逐run配置与源码hash见 evidence.json；旧版与新版前端/选优口径不同，不合并均值。Multi旧主矩阵非零λ=.05，模型/训练配置见历史catalog。

## 4. 全部主结果：evaluation

mean ± sample SD；LE为度，LR/F20为百分比。n=1不填写SD；缺少evaluation不填零。每一行按 cohort/variant 在 [runs.csv](summary_20260907/runs.csv) 可找到全部config/checkpoint/result绝对路径。

| Cohort | Variant | n scored/runs | LE_CD ↓ | LR_CD % ↑ | F20 % ↑ | ER20 ↓ | SELD_LR ↓ |
|---|---|---|---|---|---|---|---|
| audited_RB05 | C0 | 3/3 | 14.047 ± 0.288 | 72.163 ± 0.368 | 57.405 ± 0.949 | 0.627 ± 0.020 | 0.352 ± 0.008 |
| audited_RB05 | C1 | 3/3 | 13.131 ± 1.272 | 70.203 ± 1.103 | 58.082 ± 1.652 | 0.606 ± 0.029 | 0.349 ± 0.011 |
| audited_RB05 | C3_j005 | 3/3 | 13.467 ± 0.847 | 70.512 ± 0.756 | 57.534 ± 1.734 | 0.609 ± 0.040 | 0.351 ± 0.013 |
| audited_RB05 | C3_j020 | 0/3 | — | — | — | — | — |
| audited_RB05 | D1_motionpairs | 0/1 | — | — | — | — | — |
| audited_RB05 | D3_motionpairs_j005 | 0/1 | — | — | — | — | — |
| audited_rabbit02 | C0 | 3/3 | 13.622 ± 1.429 | 71.651 ± 1.031 | 57.758 ± 0.809 | 0.622 ± 0.005 | 0.351 ± 0.004 |
| audited_rabbit02 | C1 | 3/3 | 12.871 ± 1.519 | 69.510 ± 1.159 | 58.199 ± 1.389 | 0.601 ± 0.007 | 0.349 ± 0.008 |
| audited_rabbit02 | C2 | 3/3 | 13.485 ± 1.020 | 69.538 ± 1.692 | 56.567 ± 1.195 | 0.629 ± 0.006 | 0.361 ± 0.005 |
| audited_rabbit02 | C3_j020 | 3/3 | 15.330 ± 1.674 | 69.747 ± 1.116 | 54.999 ± 3.531 | 0.629 ± 0.041 | 0.367 ± 0.021 |
| historical_EINV2_causal | C0.1 | 3/3 | 13.596 ± 1.089 | 70.884 ± 0.567 | 57.538 ± 1.789 | 0.633 ± 0.046 | 0.356 ± 0.017 |
| historical_EINV2_causal | C1 | 3/3 | 14.511 ± 1.476 | 69.785 ± 1.634 | 55.646 ± 0.566 | 0.669 ± 0.041 | 0.374 ± 0.012 |
| historical_EINV2_causal | C2 | 3/3 | 14.384 ± 1.396 | 69.587 ± 1.369 | 56.213 ± 0.995 | 0.639 ± 0.002 | 0.365 ± 0.005 |
| historical_EINV2_causal | C3 | 3/3 | 13.066 ± 0.245 | 70.300 ± 1.288 | 57.441 ± 0.821 | 0.651 ± 0.019 | 0.361 ± 0.006 |
| historical_EINV2_noncausal | E0 | 3/3 | 10.286 ± 0.474 | 78.505 ± 0.261 | 72.296 ± 0.763 | 0.376 ± 0.005 | 0.231 ± 0.003 |
| historical_EINV2_noncausal | E1 | 3/3 | 10.150 ± 0.580 | 78.775 ± 1.323 | 72.960 ± 0.666 | 0.365 ± 0.005 | 0.226 ± 0.005 |
| historical_EINV2_noncausal | E2 | 3/3 | 11.692 ± 1.249 | 77.898 ± 0.770 | 70.962 ± 1.339 | 0.381 ± 0.015 | 0.239 ± 0.007 |
| historical_EINV2_noncausal | E3 | 3/3 | 12.076 ± 1.823 | 77.578 ± 0.718 | 69.936 ± 1.957 | 0.398 ± 0.020 | 0.248 ± 0.013 |
| historical_Multi-ACCDOA_causal | C0 | 3/3 | 15.344 ± 0.106 | 56.790 ± 0.879 | 48.980 ± 0.690 | 0.611 ± 0.006 | 0.410 ± 0.005 |
| historical_Multi-ACCDOA_causal | C1_lambda005 | 3/3 | 15.222 ± 0.191 | 55.805 ± 0.888 | 48.882 ± 1.015 | 0.606 ± 0.009 | 0.411 ± 0.007 |
| historical_Multi-ACCDOA_causal | C2_lambda005 | 3/3 | 15.094 ± 0.275 | 56.391 ± 0.435 | 48.790 ± 0.852 | 0.616 ± 0.014 | 0.412 ± 0.006 |
| historical_Multi-ACCDOA_causal | C3_lambda005 | 3/3 | 14.542 ± 0.155 | 54.613 ± 1.953 | 49.041 ± 1.337 | 0.611 ± 0.010 | 0.414 ± 0.011 |
| historical_Multi-ACCDOA_noncausal | A0 | 3/3 | 14.951 ± 0.365 | 62.738 ± 0.810 | 54.797 ± 0.869 | 0.521 ± 0.011 | 0.357 ± 0.007 |
| historical_Multi-ACCDOA_noncausal | A3_lambda005 | 3/3 | 14.913 ± 0.349 | 59.936 ± 1.210 | 53.061 ± 0.660 | 0.540 ± 0.008 | 0.373 ± 0.006 |

## 5. Validation 与缺失实验

| Cohort | Variant | n scored/runs | LE_CD ↓ | LR_CD % ↑ | F20 % ↑ | ER20 ↓ | SELD_LR ↓ |
|---|---|---|---|---|---|---|---|
| audited_RB05 | C0 | 3/3 | 15.169 ± 0.318 | 70.187 ± 0.589 | 55.216 ± 0.716 | 0.644 ± 0.011 | 0.369 ± 0.006 |
| audited_RB05 | C1 | 3/3 | 14.231 ± 2.472 | 68.074 ± 1.060 | 55.418 ± 3.053 | 0.632 ± 0.030 | 0.369 ± 0.020 |
| audited_RB05 | C3_j005 | 3/3 | 14.476 ± 1.930 | 69.389 ± 0.382 | 55.461 ± 1.487 | 0.634 ± 0.031 | 0.367 ± 0.010 |
| audited_RB05 | C3_j020 | 3/3 | 14.882 ± 1.928 | 66.438 ± 1.308 | 53.653 ± 1.382 | 0.636 ± 0.009 | 0.379 ± 0.007 |
| audited_RB05 | D1_motionpairs | 1/1 | 15.862 (n=1) | 68.283 (n=1) | 54.056 (n=1) | 0.656 (n=1) | 0.380 (n=1) |
| audited_RB05 | D3_motionpairs_j005 | 1/1 | 16.296 (n=1) | 68.511 (n=1) | 53.164 (n=1) | 0.665 (n=1) | 0.385 (n=1) |
| audited_rabbit02 | C0 | 3/3 | 14.210 ± 2.360 | 69.511 ± 1.002 | 55.091 ± 1.635 | 0.650 ± 0.023 | 0.371 ± 0.012 |
| audited_rabbit02 | C1 | 3/3 | 12.913 ± 1.968 | 68.674 ± 0.725 | 57.070 ± 1.439 | 0.614 ± 0.004 | 0.357 ± 0.007 |
| audited_rabbit02 | C2 | 3/3 | 13.578 ± 1.666 | 67.512 ± 1.137 | 55.136 ± 1.736 | 0.643 ± 0.022 | 0.373 ± 0.010 |
| audited_rabbit02 | C3_j020 | 3/3 | 15.497 ± 1.521 | 68.110 ± 0.878 | 54.030 ± 2.140 | 0.641 ± 0.014 | 0.377 ± 0.011 |
| historical_EINV2_causal | C0.1 | 3/3 | 14.743 ± 2.251 | 68.358 ± 0.677 | 54.636 ± 1.983 | 0.648 ± 0.036 | 0.375 ± 0.013 |
| historical_EINV2_causal | C1 | 3/3 | 15.233 ± 2.547 | 67.887 ± 0.529 | 53.922 ± 2.020 | 0.664 ± 0.008 | 0.383 ± 0.010 |
| historical_EINV2_causal | C2 | 3/3 | 15.514 ± 1.827 | 67.984 ± 0.520 | 53.732 ± 2.171 | 0.659 ± 0.023 | 0.382 ± 0.013 |
| historical_EINV2_causal | C3 | 3/3 | 12.955 ± 1.103 | 69.304 ± 1.097 | 56.911 ± 1.112 | 0.638 ± 0.004 | 0.362 ± 0.007 |
| historical_EINV2_noncausal | E0 | 3/3 | 10.226 ± 0.332 | 74.750 ± 0.674 | 68.094 ± 1.196 | 0.428 ± 0.014 | 0.264 ± 0.009 |
| historical_EINV2_noncausal | E1 | 3/3 | 10.134 ± 0.228 | 74.878 ± 0.756 | 68.396 ± 0.853 | 0.421 ± 0.007 | 0.261 ± 0.006 |
| historical_EINV2_noncausal | E2 | 3/3 | 11.306 ± 1.297 | 75.037 ± 0.593 | 67.276 ± 2.500 | 0.438 ± 0.028 | 0.269 ± 0.016 |
| historical_EINV2_noncausal | E3 | 3/3 | 11.138 ± 1.797 | 73.641 ± 0.599 | 65.471 ± 1.937 | 0.459 ± 0.020 | 0.283 ± 0.011 |
| historical_Multi-ACCDOA_causal | C0 | 3/3 | 15.721 ± 0.521 | 54.639 ± 0.122 | 46.812 ± 0.484 | 0.636 ± 0.009 | 0.427 ± 0.004 |
| historical_Multi-ACCDOA_causal | C1_lambda005 | 3/3 | 15.495 ± 0.371 | 55.196 ± 0.772 | 47.518 ± 0.440 | 0.625 ± 0.003 | 0.421 ± 0.002 |
| historical_Multi-ACCDOA_causal | C2_lambda005 | 3/3 | 15.599 ± 0.561 | 55.222 ± 0.667 | 47.379 ± 0.692 | 0.633 ± 0.006 | 0.423 ± 0.004 |
| historical_Multi-ACCDOA_causal | C3_lambda005 | 3/3 | 15.152 ± 0.349 | 53.676 ± 0.944 | 46.894 ± 0.526 | 0.633 ± 0.009 | 0.428 ± 0.004 |
| historical_Multi-ACCDOA_noncausal | A0 | 3/3 | 14.662 ± 0.477 | 59.874 ± 0.584 | 52.752 ± 1.556 | 0.548 ± 0.014 | 0.376 ± 0.008 |
| historical_Multi-ACCDOA_noncausal | A3_lambda005 | 3/3 | 14.671 ± 0.042 | 58.177 ± 1.582 | 52.024 ± 0.852 | 0.561 ± 0.006 | 0.385 ± 0.007 |

RB05 C3_j020三seeds、D1/D3 seed2026没有独立evaluation文件；不是failed test。D1/D3各有一个90epoch validation结果，但未扩到三seeds，不能据整体validation断言动态子集收益。新版smoke有2次初始化失败（错误HDF5路径），修正路径后的smoke成功，正式训练成功；保留失败记录。

## 6. 静态/动态与消融状态

历史42主run已有static/dynamic诊断：[完整分层表](EXPERIMENT_CATALOG.md#3-静态动态定位诊断)及[168条原始分层记录](catalog_20260904/completion_motion_per_run.csv)。eval GT source-frame static=64036、dynamic=53184（45.37%）；这是该分层定义下的比例，不等于moving pair比例，不能仅以“动态很少”解释性能下降。分层Recall@20是自定义逐帧指标，不是官方LR_CD。

新版EINV2的motion分层结果本次未找到；因此旧版分层收益不能直接移植到新版。新版 rabbit02 C0–C3三seed完整；RB05有C0/C1/联合两种JEPA权重，缺少JEPA-only λ=.05三seed；D1/D3仅seed2026。**更正历史概括：Multi A1/A2在旧λ=.2矩阵已有三seed，不能说只有代码；缺的是λ=.05三seed与统一重评分。**

## 7. 早期、旁支与失败记录（不删除）

[legacy_registry.csv](summary_20260907/legacy_registry.csv)列出全部检索到的旧manifest、原始评估表、训练curve的末行及最低logged selection metric行。原始数值见evidence.json；不将log最小行自动当作保留checkpoint，也不对不同metric作均值。包含旧Multi λ=.2、smoke、早期baseline，RB05旧EINV2 λ=.05，以及STARSS22/23。

### 7.1 旧Multi完整8×3矩阵（原始manifest口径）

下表不是重新对齐后的DCASE主表，故不与第4节数值直接做差。C0/A0同一run也出现在统一表中，不重复计数；列表式原始评价值中的第二项是该run内部区间，下面SD重新按三个seed计算。

| 旧Multi Variant | λ velocity / JEPA | n | LE 原始 | LR 原始 % | F 原始 % | SELD 原始 |
|---|---|---|---|---|---|---|
| A0 | 0.2/0.2 | 3 | 14.923 ± 0.372 | 62.843 ± 0.799 | 54.996 ± 0.921 | 0.356 ± 0.007 |
| A1 | 0.2/0.2 | 3 | 14.912 ± 0.217 | 61.258 ± 0.925 | 54.098 ± 1.145 | 0.365 ± 0.008 |
| A2 | 0.2/0.2 | 3 | 15.411 ± 0.432 | 56.786 ± 0.303 | 50.431 ± 0.774 | 0.393 ± 0.003 |
| A3 | 0.2/0.2 | 3 | 15.222 ± 0.458 | 55.586 ± 0.953 | 50.764 ± 1.051 | 0.396 ± 0.008 |
| C0 | 0.2/0.2 | 3 | 15.279 ± 0.056 | 56.934 ± 0.881 | 49.192 ± 0.696 | 0.408 ± 0.005 |
| C1 | 0.2/0.2 | 3 | 14.980 ± 0.604 | 53.168 ± 0.981 | 47.861 ± 0.719 | 0.421 ± 0.005 |
| C2 | 0.2/0.2 | 3 | 15.172 ± 0.467 | 53.581 ± 0.529 | 47.544 ± 1.317 | 0.423 ± 0.007 |
| C3 | 0.2/0.2 | 3 | 15.342 ± 0.101 | 50.897 ± 1.143 | 46.085 ± 0.400 | 0.435 ± 0.004 |

其余λ=.02 pilot、λ=.05、smoke的逐run validation/evaluation数值都在[original_multi_results.csv](summary_20260907/original_multi_results.csv)。

### 7.2 STARSS训练曲线（非独立evaluation）

| Dataset / seed | logged epochs | 最低logged macro-SELD所在epoch | LE_macro | LR_macro % | F_macro % | SELD_macro |
|---|---|---|---|---|---|---|
| STARSS22 / 2026 | 80 | 64 | 42.863 | 47.989 | 21.246 | 0.600075 |
| STARSS23 / 2026 | 80 | 78 | 30.934 | 43.907 | 20.605 | 0.578386 |

STARSS22/23实际有training curve，也有仅header的启动记录；未找到本次可确认的独立test结果，不能声称STARSS baseline已完整复现。ObjectStateSELD有B0工程smoke与manifest/配置，但README明确完整训练/评价及B1–B3 objectives未完成，本轮未发现正式训练成绩。PSELDNets/NTU等上游目录单列路径与Git记录，不算本方法结果。

## 8. 证据强度与设计问题

VERIFIED：主表评分文件、逐run配置、checkpoint存在/大小、90条epoch日志与完成状态、评价文件数、SELD公式重算。PARTIALLY VERIFIED：checkpoint未逐一反序列化或独立重训；旧E0 seed2026启动来源不完整；STARSS仅训练曲线。PLANNED：缺失evaluation/三seed/motion分层/JEPA-only .05。INFERRED：显式运动与latent连续性互补的机制解释，尚无稳定消融证明。

风险：LE条件于检测匹配，LE改善同时LR下降不能直接解释为全体声源定位提升；λ搜索、mask试验与多seed比较是探索性结果。每组仅3seeds/一个validation fold，不报告显著性或广泛泛化。rabbit02与RB05不混为6seeds；相同seed跨硬件不保证逐位一致。历史checkpoint按旧指标选优后重评分，与新版不同。未来teacher目标用于训练不自动构成推理泄漏；因果声明仍以当前帧边界与端到端依赖审计为准。

## 9. 下一步（本轮只整理，未启动）

1. P0：冻结新版各组best及当前表，补新版同一预测上的static/dynamic LE、匹配分母、recall，检验定位收益是否来自漏检变化。
2. P0：补齐RB05高JEPA与D1/D3现有checkpoint的独立evaluation，结果作为探索性附录，不用test继续调λ。
3. P1：若继续验证互补性，补RB05 JEPA-only λ=.05三seeds，与B0/B1/联合形成完整2×2。
4. P1：以velocity-only为主定位对照，预先固定LE/LR/F20可接受权衡；D1/D3当前无充分推广依据，先不扩展新的loss。

## 10. 代码与版本

当前代码、训练入口、评价入口、配置、服务器路径预设、历史source snapshots分层管理，见[代码地图](../docs/CODE_MAP.md)。服务器原目录不重命名、不删除、不修改；权重/数据/完整预测不上传。三个新版runtime按训练记录的逐文件SHA256恢复，防止用当前源码错评旧checkpoint。

## Evidence Index

- [全部68主run×split与绝对路径](summary_20260907/runs.csv)、[均值/SD](summary_20260907/aggregate.csv)、[配对seed差](summary_20260907/paired_seed_deltas.csv)。
- [原始配置/评分/状态/曲线/manifest快照](summary_20260907/evidence.json)、[核验结果](summary_20260907/qa.json)、[smoke/失败](summary_20260907/smoke_and_failures.csv)、[旧记录索引](summary_20260907/legacy_registry.csv)。
- [服务器目录与Git索引](summary_20260907/server_inventory.json)、[checkpoint索引](summary_20260907/checkpoints.csv)。
- [训练期runtime源文件快照](../provenance/runtime_snapshots)、[新增旧项目代码来源](../provenance/export_rabbit02_20260907.json)。
- [旧14组完整报告](EXPERIMENT_CATALOG.md)、[历史15.28→11.94的限定条件](HISTORICAL_LE.md)。

## ARS Material Passport / fallacy scan

材料：两台服务器本地实验文件，只读采集；摘要/计算由Codex生成，未经作者逐项人工确认；无USER_ATTESTED_READ声明。执行ARS experiment-agent validate，判断为Share with caveats，而非机制或论文结论已验证。

11项检查：Simpson分层待补；ecological不推断个体；Berkson/collider明确LE匹配选择；base-rate列分母；regression-to-mean保留全部seed；survivorship列失败/缺测；look-elsewhere不报筛选后显著性；forking-paths保留版本/λ；correlation≠causation不声称互补机制成立；reverse-causality不适用。
