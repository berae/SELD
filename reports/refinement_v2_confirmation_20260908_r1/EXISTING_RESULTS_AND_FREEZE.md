# 唯一 v2：已有结果、真实停止记录与冻结索引

## Material Passport

Origin Skill: academic-research-suite / experiment-agent。日期：2026-09-08。Verification Status: VERIFIED_EXISTING_RUNS_AND_FROZEN。固定 C0 seed2026，不是三套 C0。

## 先交付的已有材料

- [STABILITY_RESULTS.md](../refinement_v2_review_20260908/STABILITY_RESULTS.md)：原有 12 个学习结果、三个头 seed 的 validation 汇总及负向消融。
- [逐 run 汇总](../refinement_v2_review_20260908/stability/summary/summary.json)。原 head2026 详见 [首批结果](../refinement_v2_execution_20260907_r1/RESULTS.md)；head2027/2028 的 epochs.jsonl、COMPLETED.json、RUNNING.json 和逐录音结果在上述 stability 子目录。
- [原 PREREGISTRATION.md](../refinement_v2_review_20260908/PREREGISTRATION.md)：此前稳定性执行登记，不被本次改写。
- [本次跨 C0 预登记](CROSS_C0_PREREGISTRATION.md)：先于本轮 evaluation 生成和查看完成；新增训练仍待本次单独授权。
- [FROZEN.json](FROZEN.json)：全部权重完整 SHA256、原位置、选优/停止轮次、最后一轮真实日志、完整源码与配置快照的 607 个文件 hash。

## 实际停止 / validation 选优

| 头 seed | F-Deriv | R0 | R1 | R2 |
|---|---:|---:|---:|---:|
| 2026 | 151 / 129 | 81 / 16 | 79 / 44 | 83 / 16 |
| 2027 | 158 / 74 | 80 / 3 | 97 / 15 | 97 / 52 |
| 2028 | 161 / 120 | 86 / 14 | 83 / 3 | 84 / 14 |

表内为停止 epoch / validation-best epoch。逐次重新核对 epochs.jsonl 的严格最小 validation SELD_LR，平分取更早轮；与 best.pth 内记录和 COMPLETED.json 一致。12 次均满足双平台停止，不用末轮替换最优轮。实际规则：min40/max200、validation patience20 与 min_delta1e-5，且相邻两个10轮训练均值相对变化≤1%；不是“20轮即收敛”。

## 冻结范围

源目录保留不动。在 rabbit02 的 reports/refinement_v2_confirmation_20260908_r1/frozen 下复制封存 C0、scaler、运行时、评分器、解码/网络/损失代码、原配置，以及全部12个小头的 best.pth、实际执行配置、逐epoch日志和退出结果。权重副本设为只读，并用完整 SHA256 核对。

冻结清单 SHA256：`eb731da858558475c184cb0c8560a339d3f7cb403cd28aff495890adfc3a0f47`。

第一阶段固定15个条件：F0/F-EMA/F-KF，以及head2026/2027/2028各F-Deriv/R0/R1/R2。全量输出，不按evaluation删结果、挑seed或调参。历史evaluation已查看，不能称新盲测。原有12次训练及已有bootstrap均不重复。

CPU回归：[CPU_TESTS.json](CPU_TESTS.json)，6项通过，无优化器更新。C0 evaluation缓存200条录音完整通过，参数/缓冲区未变，首末批前缀20帧的原输出与特征扰动差均为0；该检查是有限回归测试，不是全输入空间的因果性证明。

补充只读核验：[原退出/配置/核心源码核验](ORIGINAL_GUARD_CONFIG_CORE_AUDIT.json)。12次原guardian退出码全部0、均未超时；原RUNNING.json记录的candidate.py和experiment_core.py与本次冻结版本逐hash一致。核验不更改封存清单，不重新训练或选优。

## 跨 C0 结果状态（单独计数）

| C0 seed | 固定头 seed | 已有可复用训练 | 本次新增训练 | 状态 |
|---|---:|---:|---:|---|
| 2026 | 2026 | 4 | 0 | 第一阶段评估已完成，见EVALUATION_RESULTS.md |
| 2027 | 2026 | 0 | 0 / 最多4 | 待本次明确授权与来源核验，无结果 |
| 2028 | 2026 | 0 | 0 / 最多4 | 待本次明确授权与来源核验，无结果 |

不把head2027/2028误计为C0 seed2027/2028。未完成项不填预期成绩，不将当前头seed的sample SD解释成跨C0不确定性。

本阶段完整交付：[EVALUATION_RESULTS.md](EVALUATION_RESULTS.md)。15/15条件、200条录音完成，全部检测保持及源hash核验通过，跨C0新增训练仍为0。
