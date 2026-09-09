# 前端保持冻结的首批次补核授权

2026-09-09，用户对“保持前端冻结，仅恢复主干历史标志，两份各限首批次、各60分钟，无训练”的请求明确回复“允许”。

两份baseline分别seed2027和seed2028；各一次同一首批次前向，不重跑上次五种上下文。保留主干初始化时逐参数requires_grad标志（不强行把所有参数设True），前端始终requires_grad=False；eval/no_grad，无优化器、无反向更新，保留DOA特征hook。首批输入tensor及波形文件hash、filename/segment与上次完全一致才能前向。

仅rabbit02空闲RTX3090，各独占一张，最多2并行；每份硬限3600秒，30秒进程/超时心跳。失败不自动重试。准确命令/GPU/PID记在PLAN.json、DISPATCHED和guard中。

新输出：`/work/zhanghc/Myllm/SELD/reports/baseline_frontend_frozen_check_20260909`。
上次诊断：`/work/zhanghc/Myllm/SELD/reports/baseline_repeat_diagnostic_20260909`，只读。
原模型及配置：`/work/zhanghc/Myllm/SELD/reports/refinement_v2_cross_c0_20260908_r1`，只读。

保存logits/概率/DOA/特征及相对历史和上次的差异，首批中完整录音的CSV/子集评分；不生成全validation分数。不在RB05计算，不恢复全量缓存，不训练，不放宽门槛，不覆盖历史。通过后说明恢复范围，由用户另行确认。
