# 固定C0的头seed稳定性：复用既有结果

## Material Passport

- 2026-09-09整理；原结果日期2026-09-08。academic-research-suite / experiment-agent。
- **一个固定C0 seed2026，三个小头seed2026/2027/2028**。以下不是三个独立baseline，不与本轮跨baseline配对表混计。
- 复用已冻结、已评估结果，本次对此组新增训练0、重新评估0、bootstrap重算0。evaluation非新盲测。

## 既有evaluation稳定性

主协议DCASE2023 micro，每条件200条。学习条件为三个头的均值±sample SD；固定对照只计算一次，不当作三次独立训练。

| 简单名称 | LE_CD↓（度） | SELD_LR↓ |
|---|---:|---:|
| 原模型 | 14.342918 | 0.359813 |
| 原模型＋两帧平滑 | 14.327649 | 0.359764 |
| 原模型＋卡尔曼滤波 | 14.337258 | 0.359757 |
| 原模型＋独立运动预测头 | 14.307297 ± 0.004230 | 0.359675 ± 0.000141 |
| 原模型＋方向修正 | 14.109981 ± 0.035469 | 0.358422 ± 0.000699 |
| 方向修正＋历史输入 | 14.146786 ± 0.107751 | 0.358840 ± 0.000895 |
| 方向修正＋历史输入＋运动监督 | 14.093151 ± 0.039927 | 0.357913 ± 0.000135 |

在这个固定C0上，方向修正相对原输出和独立运动预测头的LE/SELD均3/3头改善；加入历史与运动监督的逐头增量方向不一致。表中均值不能替代逐头差异，不能据此证明机制或跨C0泛化。

原15条件全部检测保持；固定匹配及静态/动态分母、逐头完整五指标、四评分协议、停止/选优记录与来源仍在下列原始材料。三个头的分布不用于给本轮三个baseline扩充独立样本量。

## 原始来源（不重写历史）

- [既有evaluation全量结果与分层诊断](../refinement_v2_confirmation_20260908_r1/EVALUATION_RESULTS.md)。
- [既有逐头配对差与sample SD](../refinement_v2_confirmation_20260908_r1/HEAD_SEED_DESCRIPTIVE_QA.json)。
- [既有validation稳定性和逐run停止轮](../refinement_v2_review_20260908/STABILITY_RESULTS.md)。
- [原evaluation SUMMARY](../refinement_v2_confirmation_20260908_r1/evaluation/SUMMARY.json)，SHA256 `f45b885d024c91dff9949a9f16cac2bd93b81eef58295d81459b4b67687bfa49`。
- [原FROZEN](../refinement_v2_confirmation_20260908_r1/FROZEN.json)，SHA256 `eb731da858558475c184cb0c8560a339d3f7cb403cd28aff495890adfc3a0f47`。

技能影响：只复用已核验材料，明确重复单位与历史evaluation暴露；不把平均值优势等同于每seed改善或显著性证明。
