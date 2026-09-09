# 两份 baseline 首批次差异诊断

## Material Passport

- 日期：2026-09-09；Origin Skill: academic-research-suite / experiment-agent；Origin Mode: authorized run。
- Verification Status: DIAGNOSTIC_COMPLETED_NOT_CACHE_RELEASE。两份诊断各执行一次，均退出0、未超时、未重试。新增训练仍0/8。
- 用户将每份硬限由30分钟改为60分钟；实际诊断约20秒，guard约30秒收集退出。只用rabbit02空闲3090，没有RB05计算。
- 本文是诊断与下一步边界，不是完整validation回归通过，也不是新的方法结果。

## 结论先行

当前冻结标志的导出路径在两份baseline首批上复现了浮点不一致。将参数requires_grad标志设为True、同时仍在no_grad下执行，可使同一3090上的SED logits、概率和DOA与历史参考逐位一致。保留特征hook仍可一致；去掉hook本身不能消除当前差异。

**关键限定：诊断中设True的对象同时包括主干和前端，而历史Frontend构造器在runtime.py中明确调用requires_grad_(False)。因此代码/JSON中`historical_flags_*`只是诊断分支标签，不能按字面声称精确恢复了全部历史标志。** 当前结果定位到“参数标志组合”的执行差异，不足以独立归因于主干或某个底层内核。要确定最小恢复方式，还需保持前端原始冻结状态、仅恢复主干初始化标志的有界核验；本次没有自动追加第六次前向或重写原结果。

先前进度消息曾简写为“恢复历史参数标志”，以本段的准确限定为准。没有源码/权重更新，没有优化器或反向传播；requires_grad标志为True不等于本次发生训练，结束时标志恢复False，参数及缓冲区hash未变。

## 覆盖范围

每份baseline只取原validation生成器的第一个batch，共32个4秒chunk：第一、第二条录音各15个chunk，第三条录音前2个chunk。未运行第二个batch。

完整录音是`fold1_room1_mix001_ov1`和`fold1_room1_mix002_ov1`，另有`fold1_room1_mix003_ov1`局部浮点输出。官方评分仅比较前两条完整录音，绝不是100条validation总成绩。

五种受控导出上下文是诊断手段，不是新增实验方法：冻结标志有/无hook、参数标志设True有/无hook、冻结标志加inference_mode。训练网络、损失、解码规则均未修改。

## 首批浮点与解码结果

| baseline | 当前导出最大logit差 | 最大概率差 | 最大DOA分量差 | 参数标志设True后各字段差 |
|---|---:|---:|---:|---|
| seed2027 | 2.0980835e-5 | 4.1723251e-7 | 5.9604645e-7 | logits/概率/DOA均逐位相同，最大差0 |
| seed2028 | 2.0980835e-5 | 2.8312206e-7 | 7.1525574e-7 | logits/概率/DOA均逐位相同，最大差0 |

DOA分量差不是角度误差。当前导出的logits不等元素分别32,465/35,840与32,188/35,840；不因数值小而豁免严格门槛。

- 五种上下文的首批活动变化、活动类别变化均为0。
- 所有上下文的两条完整录音CSV均与历史逐字节一致、有序检测条目一致；各baseline分别440/290条、419/255条检测记录。
- 两条录音子集的四种既有评分协议、全部字段差均为0。不能由此推出其余98条也无变化。
- 有hook时，参数标志设True与当前导出的DOA头前特征均逐位一致：每份1,310,720个元素，最大差0。
- 冻结参数时去掉hook、或切换到inference_mode，均与当前导出浮点相同，不能修复此首批差异。

完整逐字段数量、误差分布、逐录音CSV/评分及来源见[seed2027](artifacts/baseline_2027/RESULTS.json)、[seed2028](artifacts/baseline_2028/RESULTS.json)。

## 对旧缓存和最小修复的影响

1. 首批特征一致不等于全部train特征一致，更不等于原DOA、概率、历史关联及GT固定目标侧文件一致；现有两份train缓存继续保留且不放行拟合。
2. 最小待核验改动应是导出上下文：保持历史前端冻结，主干保留历史初始化参数标志，同时强制eval/no_grad、零优化器、参数/缓冲区hash不变。暂未实施生产缓存器修复，不更换权重、参考或比较精度。
3. 补充核验若支持该修复，恢复也需独立新目录；完整validation的浮点/CSV/评分回归必须全部通过。旧train内容应在同一导出语义下核验；如原概率/DOA变化，需要按新语义重建相关缓存/关联/固定目标，不能只拼接相同h后假装全部可复用。
4. 当前未恢复导出、未启动四种小头、未查看新evaluation。原两份baseline×四头、headseed2026、40–200轮双平台、每头4小时、最多4并行、八头全部冻结后评估的授权范围不变。

## 执行与来源

- [授权（60分钟）](AUTHORIZATION.md)、[精确命令和GPU分配](artifacts/PLAN.json)、[总退出](artifacts/EXIT.json)。
- [seed2027 guard](artifacts/guards/baseline_2027/EXIT.json)、[seed2028 guard](artifacts/guards/baseline_2028/EXIT.json)：exit0、无超时/重试，单独保留PID、开始/结束时间和30秒心跳。
- source root：`/work/zhanghc/Myllm/SELD/reports/refinement_v2_cross_c0_20260908_r1`。源目录不是Git仓库，不能虚构远端HEAD；实际执行依封存源码/配置hash。发布副本基线提交为dc765371ab5b573a932d3752046aed3468933d2d。
- 新目录：`/work/zhanghc/Myllm/SELD/reports/baseline_repeat_diagnostic_20260909`。完整浮点/特征数组留在此处，不向Git上传。
- 诊断脚本SHA256：`b17ccd1a3b5e2432850d510e26f3cc48ed5aaffeb3fb7c4489c1405863fe6466`；调度脚本SHA256：`0601774c5b15b01faa0ed26776b43973a76f2251daacc7eaa4778bccd32816df`。源码副本在artifacts/code，未经事后改写。
- 本地元数据包与远端SHA256相同：`a8009e70b969c79759c73a9881d02aef13ab5db31832c5407cdd22f4cba4c8ff`。每份17个非数组产物与OUTPUT_HASHES核对通过，6个数组文件仅留远端；包括子集CSV和全部对照JSON。
- 归档时tar提示根目录mtime变化（归档文件创建于该目录且已排除自身）；包可完整列出/解压，所需34个非数组文件均经来源hash校验。没有借归档警告重新运行诊断。
- 首次远端语法检查命令有引号转义错误，未执行任何模型；改正命令后两脚本静态解析通过，正式模型诊断各只启动一次。
- 两张GPU在任务退出后释放，核查时5张3090均0MiB。不是持续空闲保证。

技能约束使本轮停在诊断交付：不自动补跑、不将首批一致推广为全量一致、不以诊断上下文的宽泛命名冒充严格历史复现。下一步需要作者确认前端保持冻结的有界补充核验，然后再决定缓存修复/恢复。
