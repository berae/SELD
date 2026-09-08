# 唯一 v2：跨 C0 确认已获授权，阻塞于原始 validation 回归

## Material Passport

日期：2026-09-08。Origin Skill: academic-research-suite / experiment-agent。Verification Status: BLOCKED_AT_VALIDATION_FLOAT_REGRESSION。当前阶段：授权→来源核验→缓存导出；尚未进入小头训练。不是新方法提案。

## 本次授权及已完成事项

- [AUTHORIZATION.md](AUTHORIZATION.md)：用户本次“确认”明确授权最多8个小头及必要的RB05只读复制；绑定前一轮已经封存的跨C0登记，不由旧授权推定。
- 两份C0权重、配置、各100份历史validation浮点/CSV参考已从RB05复制到rabbit02；不在RB05推理或训练。传输细节见[TRANSFER_NOTES.md](TRANSFER_NOTES.md)。
- C0 seed2027权重SHA256：`1f924b811f8cc8ed3cfde60d9727f54bb83e950b47404cfa09bcbc694c8aa08c`。
- C0 seed2028权重SHA256：`5d128ad298b110d3afff8ae95fc37b08d25e38fb9d1f595e8d51d99708c76683`。
- 接收端权重、checkpoint内配置、同cohort配置、运行时与scaler核验通过；200份历史浮点参考全部通过自身原manifest的hash核验。不是用rabbit02另一批C0替代。
- [PREPARED.json](PREPARED.json)保留源文件、代码、配置及本次授权hash。仅改变C0身份与存储路径，candidate.py、experiment_core.py、train_heads.py与上一阶段封存版本完全一致。
- [CPU_TESTS.json](CPU_TESTS.json)6项通过；[DIAGNOSTIC_IO_TEST.json](DIAGNOSTIC_IO_TEST.json)确认NPZ预加载优化不改变数组或固定目标误差。所有这些测试均无优化器更新。

## 明确配置

| C0 | 小头seed | 配置 | 本次正式训练数量 |
|---|---:|---|---:|
| 2027 | 2026 | [pilot_C0_2027.json](configs/pilot_C0_2027.json) | 0/最多4 |
| 2028 | 2026 | [pilot_C0_2028.json](configs/pilot_C0_2028.json) | 0/最多4 |

[execution_cross_c0.json](configs/execution_cross_c0.json)是本次实际执行覆盖层：F-Deriv/R0/R1/R2，AdamW lr=.001、weight_decay=.0001、batch32、FP32、clip1；min40/max200、validation patience20/min_delta1e-5及相邻10轮训练均值相对变化≤1%的双平台停止；严格最小validation SELD_LR选优，平分取早；每头4小时硬限，最多4并行。原pilot内20epoch与“未授权”等旧状态是保留的历史提案字段，不替代该实际执行层和本次授权回执。

无结构、loss、lambda、解码阈值或数据集调整；现有12个头和已有bootstrap均未重跑。

## 本轮四个缓存任务的实际结果

| C0 | train缓存 | validation缓存 | 新小头 | 新evaluation |
|---|---|---|---:|---|
| 2027 | 500/500完成，退出0，参数/缓冲区未变 | 首条SED浮点回归失败，退出1，写入0/100 | 0 | 未开始 |
| 2028 | 500/500完成，退出0，参数/缓冲区未变 | 首条SED浮点回归失败，退出1，写入0/100 | 0 | 未开始 |

两份train导出的首末批前20帧波形扰动回归，原预测和特征差均0；但这不能替代失败的validation原始输出核验。train缓存保留，**未放行用于拟合**。原4项缓存各只执行一次，未自动重跑。两个已启动的train导出正常完成后，全部本轮计算进程已结束，5张GPU均恢复0MiB。

## 失败证据与当前边界

两个validation任务均在`fold1_room1_mix001_ov1`的`sed`字段触发：

```
AssertionError: ('fold1_room1_mix001_ov1', 'sed', 'original raw mismatch')
```

该字段为原模型SED浮点输出，严格逐位比较未通过。断言发生在首条新浮点/CSV落盘之前，因此目前**没有可报告的误差幅度、概率/DOA比较、CSV差异或官方指标差异**。不能声称“只是微小舍入误差”，也不能据此声称检测类别已经变化。未放宽`np.array_equal`，未修改评分标准。

[READ_ONLY_FAILURE_AUDIT.json](READ_ONLY_FAILURE_AUDIT.json)进一步确认：

1. 两份实际权重与各自历史manifest一致，checkpoint配置一致。
2. runtime源码、GT文件hash一致；本次validation使用的rabbit02波形HDF5 hash，与此前已通过的C0seed2026 validation缓存相同。这不是对RB05未记录波形hash的额外证明。
3. 历史运行设备A6000，本次3090；PyTorch均记录为2.4.1+cu121。
4. 历史C0seed2026 manifest指向export.py，seed2027/2028指向export_v2.py。读取的现存脚本中，前者使用inference_mode并冻结requires_grad，后者使用no_grad。当前缓存器是no_grad并冻结requires_grad、增加DOA特征hook。脚本/上下文差异与跨设备差异是**待验证线索，不是已确定根因**；历史manifest没有封存导出脚本hash，现存源码也不能单独证明历史执行字节。

读取脚本留档：[historical_export.py](historical_export.py)、[historical_export_v2.py](historical_export_v2.py)。只读核查没有重新执行模型前向或训练。

根据预登记及experiment-agent失败不自动重跑规则，暂停于此。建议下一步另行授权**rabbit02上每个C0仅首批次的差异定位**，保留原始logits/概率/DOA/CSV比较，无训练、无门槛放宽；诊断结果出来后再决定是否修复和重跑失败validation缓存。当前不自动执行这一诊断，也不恢复正式训练。

## 两类结果仍独立计数

- 固定C0头seed稳定性：仍为此前C0seed2026×3头seed×4条件，已有12次训练和完整15条件evaluation，见[上一阶段结果](../refinement_v2_confirmation_20260908_r1/EVALUATION_RESULTS.md)。不修改已冻结结果。
- 跨C0权重配对：C0seed2026/head2026的4结果可复用；C0seed2027/2028均无新增头成绩。当前新增训练0/8，跨C0已完成权重组仍为1/3，不填预计指标、不计算跨C0稳定性。

## 归档

完整失败日志、进程回执、来源与缓存清单、授权和实际配置见[BLOCKED_ARTIFACTS_NO_WEIGHTS.tar.gz](BLOCKED_ARTIFACTS_NO_WEIGHTS.tar.gz)，已解包至artifacts目录。传输两端hash一致：`15ac3e70dfa623ea5b78f4fee7c411a49a10918770a73474ad7190715b2ea2b2`。原始C0权重、参考预测、完整train特征及部分传输文件继续保留rabbit02，不删除或覆盖原始结果。

该技能影响了执行决策：失败回归被保留为硬门槛，未把来源hash通过误当作预测回归通过，未以旧授权自动启动修复重跑。

代码上传：上一阶段842b563与本阶段95d7620均已推送GitHub main。本阶段新增小头训练、训练后冻结及evaluation路径尚未执行；仅来源准备、CPU测试、缓存导出和失败只读审计有执行证据，不把“代码已上传”写成“实验已通过”。
