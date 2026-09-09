# SELD实验状态更新（2026-09-09）

## Material Passport

- academic-research-suite / experiment-agent；授权执行及描述性汇总。
- 本日新增范围仅唯一v2的两份baseline重复验证；历史主干矩阵不重训、不重复审计或重评分。

## 当前完成情况

| 层级 | 当前记录 | 计数边界 |
|---|---|---|
| 历史主干实验 | 沿用既有总览记录的68个run及136个run×split | 本轮新增主干训练0；不与小头合成同质重复数 |
| 固定C0的头seed稳定性 | 原12个小头、15个evaluation条件 | 一个C0、三个头seed，本轮全部复用 |
| 两份baseline重复验证 | 新增8个小头全部双平台收敛、14个evaluation条件全部完成 | 两个既有C0，各四独立head seed2026 |
| 跨C0配对表 | 三个baseline×七条件=21行 | 原baseline七条件复用；不是新增21次训练 |
| 唯一v2累计 | 20个已训练小头；29个不同evaluation条件，每条件200录音 | 不把缓存、预检、重评分或两张有重叠的表重复计数 |

主结论：方向修正相对原输出，在三个baseline上LE/SELD均改善，平均差分别−0.203992°、−0.001517。显式历史增量不稳定；运动监督相比历史版在两个baseline改善、一个略退步。历史evaluation已暴露，不是新盲测；不换候选、不另起方向。

## 权威入口

- [两份baseline完整结果与结论](baseline_repeat_recovery_20260909/RESULTS.md)。
- [跨baseline完整配对表](baseline_repeat_recovery_20260909/paired_results/PAIRED_RESULTS.md)。
- [固定C0头seed稳定性，单独呈现](baseline_repeat_recovery_20260909/FIXED_C0_HEAD_STABILITY_REUSED.md)。
- [全部历史实验总览（保留9月8日时点）](CURRENT_EXPERIMENT_SUMMARY_20260908.md)。其中跨C0“0/8、未完成”已由本日完成结果替代，不回写历史记录。

新增头仅用rabbit02空闲3090，无RB05计算；完整validation回归、检测保持、固定分母与因果检查、逐epoch停止/选优和来源hash均交付。旧缓存失败、静态预检断言失败及SSH连接中断保留；没有重训任何已完成头，没有重算既有bootstrap。大权重/NPZ留rabbit02，仓库收轻量结果和源码。
