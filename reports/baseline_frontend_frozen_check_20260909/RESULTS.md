# 两份 baseline：前端保持冻结的补核结果

## Material Passport

- 日期：2026-09-09；Origin Skill: academic-research-suite / experiment-agent；Origin Mode: authorized run。
- 状态：首批次补核通过；不是全量缓存恢复通过。新小头仍0/8。
- 两份baseline各执行一次前向，硬限各3600秒，实际诊断约9.3秒，guard约30秒收齐退出。均exit0、无超时、无自动重试。

## 1. 明确结论

**保持前端冻结、仅恢复主干初始化时的参数标志后，两份baseline首批的原始输出均与历史参考逐位一致。** 上一轮同时切换前端标志的限定已由本次单独控制补齐。

主干171个参数张量的原始requires_grad标志均为True，前端3个参数张量始终False；逐参数标志记录在INPUT_MANIFEST.json。全程eval/no_grad，无优化器、无反向传播、无参数更新，前后主干及前端的参数/缓冲区hash完全一致。这里恢复标志是为了复现导出执行路径，不是放开主干训练。

输入tensor、三个实际波形HDF5文件、filename/segment顺序均与上一轮诊断hash/内容一致。使用同一台rabbit02的空闲3090，没有RB05计算。结果支持最小导出修复，不支持将此前问题归因于“跨GPU必然无法逐位一致”；没有追踪到底层具体算子/内核，故不作该层面的归因。

| 核验 | baseline seed2027 | baseline seed2028 |
|---|---|---|
| SED logits与历史参考 | 35,840元素逐位相同，最大差0 | 35,840元素逐位相同，最大差0 |
| 检测概率与历史参考 | 35,840元素逐位相同，最大差0 | 35,840元素逐位相同，最大差0 |
| 原始DOA与历史参考 | 7,680元素逐位相同，最大差0 | 7,680元素逐位相同，最大差0 |
| DOA头前特征与上次冻结导出 | 1,310,720元素逐位相同 | 1,310,720元素逐位相同 |
| 首批活动/活动类别变化 | 0 / 0 | 0 / 0 |
| 首批内两条完整录音CSV | 与历史逐字节一致 | 与历史逐字节一致 |
| 两条录音子集官方评分 | 四种协议全部字段差0 | 四种协议全部字段差0 |
| 参数与缓冲区 | 未改变 | 未改变 |

覆盖依然只是首个32-chunk batch：两条完整60秒录音和第三条录音前两个chunk。不运行第二批，不将两条子集评分当100条validation成绩。未重跑既有头、bootstrap或任何新方法。

## 2. 最小恢复范围（尚未执行，待用户确认）

1. 在两份baseline的**新导出路径**中，主干保留历史初始化参数标志；前端继续冻结。仍使用eval/no_grad，确保无优化器、无grad，前后state hash不变。只改导出执行上下文和独立输出路径，不改模型、权重、网络、损失、解码、scaler或参考。
2. 两份baseline的train/validation缓存均在同一修复后的导出语义下重新生成，各500/100条；完整validation必须通过既有严格浮点/CSV/评分回归，不放宽np.array_equal。比较新旧train特征/概率/DOA与派生目标，保留差异清单。虽然首批h相同，但旧DOA/概率已有不同，不能直接放行旧500条全部缓存或混用其关联/目标。
3. 保留所有旧缓存和失败记录，不覆盖、不删除；固定baseline seed2026的既有权重、缓存、结果不改。
4. 在来源、完整回归、缓存一致性及无更新预检均通过后，才可接续原授权的8个独立小头：原模型＋方向修正、再加历史输入、再加运动监督、独立运动预测对照。小头seed统一2026；原输出、两帧平滑、卡尔曼对照保持。
5. 训练仍至少40/最多200轮、双平台停止、validation-best、每头4小时、最多4并行，仅rabbit02空闲3090；八头全部冻结后再评估，不按evaluation调整配置或挑seed。缓存任务硬限沿用原登记30分钟；本次60分钟只属于诊断，不自动扩大其他预算。

原8头授权无需重新推定或扩充；**本次回复“允许”仅授权这次补核，尚未授权生产缓存器修复和全量恢复**。没有在补核完成后自行启动后续任务。

## 3. 交付和来源

- [授权](AUTHORIZATION.md)、[精确命令/GPU](artifacts/PLAN.json)、[总退出](artifacts/EXIT.json)。
- [seed2027完整结果](artifacts/baseline_2027/RESULTS.json)、[输入和参数标志](artifacts/baseline_2027/INPUT_MANIFEST.json)、[guard退出](artifacts/guards/baseline_2027/EXIT.json)。
- [seed2028完整结果](artifacts/baseline_2028/RESULTS.json)、[输入和参数标志](artifacts/baseline_2028/INPUT_MANIFEST.json)、[guard退出](artifacts/guards/baseline_2028/EXIT.json)。
- 远端：`/work/zhanghc/Myllm/SELD/reports/baseline_frontend_frozen_check_20260909`；源码、授权、日志与全部浮点数组在内，不改写上轮目录。
- 诊断代码SHA256：`dfd4b1d548aa93d3f6a0e86a1a2549e515d2719dbbc8f35659b0e7896328a106`；调度代码SHA256：`51cd4be1ab915fbbb4a99eadf7adcb6bc59dfbd33c3751d01b8101ab4700e51d`。执行前静态语法解析通过，传输两端一致；每项还重新核验原封存代码、配置、checkpoint/scaler及所用历史参考hash。
- 元数据包SHA256：`c170b7a069c1d11e94edfbed9e3bc0ae597ba25a41a2fe1ab68e7eb11f2b11ca`，本地与远端一致；两份各5个非数组产物与OUTPUT_HASHES逐个校验通过。数组留远端，仓库只收轻量记录和4份子集CSV。
- GPU任务退出后已释放，核查时5张3090均0MiB；启动后续任务时必须再次检查空闲状态。

技能影响：保留授权边界、原始记录和严格门槛；不把首批通过升级为全数据通过，不以模型标志为True冒充主干更新，也不把恢复建议当成已经执行的结果。
