# 两份 baseline 首批次诊断授权

日期：2026-09-09。用户在接手“两份 baseline 重复验证”后，明确回复“允许”，同意rabbit02空闲3090上两份baseline各一次首批次差异诊断；随后要求“时间改为60分钟吧”。

- 每份baseline的诊断硬限为3600秒；最多两份同时，各独占一张空闲RTX3090。
- 仅seed2027与seed2028已有准确权重，固定同一个首批次32个chunk。对比当前与历史导出的参数标志、hook、no_grad/inference_mode上下文；不是新增训练方法。
- 五种预先固定上下文：冻结参数+hook+no_grad；冻结参数+无hook+no_grad；历史参数标志+无hook+no_grad；历史参数标志+hook+no_grad；冻结参数+hook+inference_mode。每种仅同一批次一次前向，不读取下一批次用于推理。
- no_grad/inference_mode下不构建反向图、不创建优化器；临时恢复历史requires_grad标志不更新参数，前后参数/缓冲区hash核对。
- 保存原始检测logits/概率、方向浮点、可用特征和差异；对首批内完整录音保存CSV与同子集官方评分。子集评分不冒充全validation。
- 不在RB05计算或读取新增材料，不恢复全量缓存、不训练、不放宽门槛、不替换历史参考、不覆盖旧目录。
- 原始波形不另存、不上传；输出数组与旧参考摘录只保留计算服务器；小型报告可按已有仓库要求归档。
- 失败不自动重试；诊断完成后报告最小修复范围和旧train缓存影响，修复/恢复须另行确认。

源目录：`/work/zhanghc/Myllm/SELD/reports/refinement_v2_cross_c0_20260908_r1`。
新诊断目录：`/work/zhanghc/Myllm/SELD/reports/baseline_repeat_diagnostic_20260909`。
精确入口和设备在该目录PLAN.json与逐baseline DISPATCHED记录，未创建旧实验恢复任务。
