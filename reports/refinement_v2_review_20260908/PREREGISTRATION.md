# 2026-09-08 补充检查登记

## Material Passport

- Origin Skill: experiment-agent / validate-data
- Origin Mode: validate + explicitly authorized run
- Origin Date: 2026-09-08
- Verification Status: UNVERIFIED (registration, not results)
- Version Label: fixed_C0_head_stability_r1

用户允许按分析需要在rabbit02补实验。只追加小头初始化/样本顺序敏感性：固定原C0 seed2026，四个学习条件各补head seed2027、2028；与原head seed2026合并描述。不是三个独立主干seed，不是盲测，也不更换唯一v2。

全部网络、loss、lambda、数据、完整验证、最小SELD选优和双平台规则不变。每次最多四张无其他计算进程的3090，每个训练4小时硬限；监测存活/日志，失败保留且不自动重跑。2027全部正常退出后再检查空卡并启动2028；无空卡则停止排程。

预定比较全部报告：R0−F0、R0−F-Deriv、R1−R0、R2−R1。主指标SELD_LR，LE_CD共同报告；全部seed均值、sample SD和配对差，不选择最好seed。保留原单seed结果，不做显著性或独立泛化声明。

追加CPU分析：从逐录音官方计数重构总分；核对100条一致覆盖；训练最优/末轮对比；按录音重叠标记分组并重新聚合分子分母，分开报告录音均值与事件加权固定目标均值；不把34314帧当独立实验。不新增阈值搜索或方法候选。
