# 两份 baseline 重复验证：缓存恢复完成，预检暂停

## Material Passport

- 日期：2026-09-09；Origin Skill：academic-research-suite / experiment-agent；Mode：authorized run。
- 执行状态：缓存恢复完成；无更新预检失败；小头训练 **0/8**，新增 evaluation **0/14**。不是最终实验结果。
- 本次用户已允许修复导出并恢复全量缓存，通过全部门槛后继续原8头。本次新预检失败后没有自动重跑、修改门槛或启动训练。
- 唯一 v2，模型/损失/训练核心不变，无主干重训、无新方法、无旧头或bootstrap重算。历史evaluation已暴露。

## 已完成

| baseline | train缓存 | validation缓存 | 原始浮点／CSV／四协议评分回归 | 新小头 |
|---|---:|---:|---|---:|
| seed2027 | 500/500，134.77秒 | 100/100，43.54秒 | 完整validation严格通过 | 0/4 |
| seed2028 | 500/500，134.53秒 | 100/100，43.58秒 | 完整validation严格通过 | 0/4 |

表中时间是导出器记录，不含guard轮询收尾；逐任务准确值以COMPLETED与guard为准。四任务exit0、无超时、无重试。两份baseline首尾导出批次的raw/feature前缀差均为0；检查范围是所测批次，不是全输入因果证明。模型及前端参数/缓冲区前后不变，无梯度累积、无优化器。

修复仅恢复历史模型默认参数标志，前端仍冻结，eval/no_grad；`requires_grad=True`不等于执行训练。默认导出行为未全局改变，仅本次2027/2028显式启用。旧缓存及失败记录完整保留。

## 新旧训练缓存全量比较

两份各核验2000个文件hash并比较数组内容，实际波形、GT一致。**所有拟合仅允许使用新缓存，不混用旧缓存。**

| 字段 | seed2027 | seed2028 |
|---|---|---|
| 隐藏特征 | 499/500条逐位一致；1条428212元素不同，最大差2.8610e-6 | 499/500条逐位一致；1条431276元素不同，最大差3.4571e-6 |
| 原SED logits | 500条有差异，最大差2.8610e-5 | 500条有差异，最大差3.0518e-5 |
| 检测概率 | 最大差5.9605e-7 | 最大差7.4506e-7 |
| 原始DOA | 最大差7.1526e-7 | 最大差1.0133e-6 |
| 原始CSV | 2条字节不同 | 6条字节不同 |
| 预测历史关联 | 全部相同 | 全部相同 |
| 固定目标、GT身份、匹配mask、有效motion mask、位移 | 全部相同 | 全部相同 |

此比较不是通过容差放行旧缓存，也不是要求新缓存复现已知错误的旧导出上下文。完整历史validation参考仍逐位通过。首批诊断中隐藏特征相同不能推广到全train，本次已如实记录全量差异，并完成两份train整体重建。

## 新暂停点：预检样本不适合“非零梯度”断言

两份预检均退出1，无超时；失败位置为 `preflight_cross_c0.py:39`，末层权重梯度绝对值之和必须大于0。没有生成PREFLIGHT通过回执，也没有创建heads。

按代码顺序，首次检查对象是“原模型＋独立运动预测头”。预检固定取首条训练录音 `fold2_room1_mix001_ov1.csv` 的前两个4秒chunk，共80帧。失败后只读检查已保存的目标，不再运行模型：

| 证据 | seed2027 | seed2028 |
|---|---:|---:|
| 前80帧固定匹配 | 61 | 61 |
| 前80帧有效motion pair | 59 | 59 |
| 这些pair的非零位移坐标数 | 0 | 0 |
| 这些pair的位移最大绝对值 | 0 | 0 |
| 同录音全部600帧有效motion pair | 452 | 453 |
| 同录音全部有效pair的非零位移坐标数 | 324 | 324 |

目标文件SHA256：

- seed2027：`bc637c5b79d3454a70e5812f6c429981cfddb2763b6c2a558212b6f9e33beb0d`
- seed2028：`81e638bdb5912aaed7d5113aaa5d3c7803956d1ea95fbb1db42c734cc10c8780`

代码使用零末层初始化；独立运动预测头输出因此为0。选中目标也全为0，SmoothL1项与零输出平方正则的梯度均为0。这解释当前断言失败，**不是训练不收敛的证据，也未证明完整梯度链路通过**。不把后续三个头的梯度预检记为通过。

## 待用户确认的最小预检修正（未执行）

1. 保留前80帧作为静态零目标测试：独立运动预测头应为零损失、零梯度且有限。
2. 非零梯度测试改用**同一首条训练录音全部15个chunk**，保持每chunk独立重置；已有目标证明其中包含非零运动信号。不搜索validation/evaluation，不依评分挑样本，不改训练数据。
3. 仅改预检夹具和记录：固定预检随机种子、记录每个头的loss/梯度和目标分母；保留“末层梯度有限且非零”的正样本门槛。模型、损失、训练、选优代码完全不变。
4. 新独立预检输出目录，保留本次失败记录及已产出的固定对照，不覆盖、不重新导出缓存；两份各一次有界CPU无优化器预检，30分钟硬限，无自动重试。
5. 全部通过后再放行已授权8头；每头仍4小时、至少40/最多200轮、双平台停止、独立head seed2026。八头冻结之后才新evaluation。

这个建议不是已经完成的修复或通过证明。本轮按技能失败不自动重试规则暂停，等待用户确认此精确范围。

## 产物与来源

- 远端根目录：`/work/zhanghc/Myllm/SELD/reports/baseline_repeat_recovery_20260909`。
- [本次授权](AUTHORIZATION.md)、[来源封存](PREPARED.json)、[新旧训练缓存全量差异](TRAIN_CACHE_COMPARISON.json)。
- [四任务调度](artifacts/dispatch/cache_both_C0/PLAN.json)、[总退出](artifacts/dispatch/cache_both_C0/EXIT.json)。
- [seed2027预检日志](artifacts/guards/preflight_2027/process.log)、[退出](artifacts/guards/preflight_2027/EXIT.json)；[seed2028预检日志](artifacts/guards/preflight_2028/process.log)、[退出](artifacts/guards/preflight_2028/EXIT.json)。
- `artifacts/C0_2027/cache`、`artifacts/C0_2028/cache`：完整缓存输入/文件manifest、固定目标分母、完成记录及原输出CSV；浮点数组和权重只在远端，不放仓库。
- `artifacts/C0_2027/validation_controls`、`artifacts/C0_2028/validation_controls`：已产出的原输出／两帧平滑／卡尔曼对照及逐录音评分；不是新evaluation。
- 元数据包SHA256：`9718ca8e5133a2d10d112e6a82fac20263057da0a4d472bd464484d15b79b682`，本地与远端一致。
- 传回的1200份原输出CSV逐个与缓存manifest核验通过。元数据包另保留600份validation固定对照CSV，共1800份CSV；不含权重、NPZ或GT原文件。
- 修复代码及此前两轮诊断已推送main `1b426ed`。本状态与产物在后续提交上传；以实际push回执为准。

当前没有新学习结果可作baseline配对差、收敛表或evaluation检测保持结论，不用预计值填表。原固定C0头seed稳定性结果不受本次操作修改，也不与本次0/8混计。
