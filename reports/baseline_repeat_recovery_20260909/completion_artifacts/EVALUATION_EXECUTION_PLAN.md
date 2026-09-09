# 全部八头冻结后的评估与交付计划

## Material Passport

- 2026-09-09；academic-research-suite / experiment-agent，run。
- 沿用原授权和唯一v2，实施层登记，不改旧预注册。本文件在新evaluation导出/评分之前写入。
- 训练：两份baseline各四头、head seed2026，各自独立初始化；已有baseline2026结果仅复用。

## 顺序与固定设置

1. 两份四头训练均完成后，核对逐guard退出和逐头COMPLETED；仅成功完成的原进程证据可形成总回执。不得重训或把传输中断当作重启理由。
2. 用已核验的 `code/refinement_v2/freeze_cross_c0.py --root <根>` 逐一核对validation选优、停止记录和权重hash，生成八头总封存回执 `ALL_EIGHT_FROZEN.json`。新evaluation仍未打开。
3. `operations/start_registered_stage.py --root <根> --stage evaluation-cache`：入口再次核验八头总回执及每份FROZEN全部文件hash，然后为两份baseline各导出evaluation200。仅空闲3090，每任务1800秒，保留修复后的历史模型标志、冻结前端和no_grad，参数值不变。
4. 两份缓存均完成后，`operations/start_registered_stage.py --root <根> --stage evaluation-score`：各自原输出、两帧平滑、卡尔曼、独立运动预测头、方向修正、加历史、加运动监督，共14条件，CPU FP32，无优化器、无按分数调参或选seed；每任务1800秒。
5. `operations/summarize_cross_baselines.py --root <根> --previous /work/zhanghc/Myllm/SELD/reports/refinement_v2_confirmation_20260908_r1 --output <根>/paired_results`：仅复用原baseline/head2026七条件，得到三份baseline各七条件。完整保留四套官方评分，主表DCASE2023 micro；原始精度JSON、每录音CSV/评分、固定匹配及静态/动态诊断、检测保持、因果测试、来源hash均保留。

根目录：`/work/zhanghc/Myllm/SELD/reports/baseline_repeat_recovery_20260909`。所有计算在rabbit02；Python与环境不变。各任务实际命令、PID、GPU UUID、开始/退出记录由PLAN/guard保存。

## 汇总边界

- 配对：方向修正−原输出、方向修正−独立运动预测头、加入历史−仅方向修正、加入运动监督−加入历史。
- 同baseline固定匹配分母；不同baseline分母如有不同如实保留。静态/运动pair与整段source分层不混合，不按20°筛样或事后重新匹配。
- 三份配对差完整后才计算等权mean与sample SD，不作新bootstrap、置信区间或p值，不填预计结果。
- 原固定C0的三个头seed稳定性仍单列，不能把三baseline与三头seed混成独立重复数。
- 历史evaluation已经查看，不能称新盲测。一个数据集、同一架构、三个权重，不推广成跨数据/架构泛化；未增加等容量对照，不对历史输入或运动监督作超出消融的机制归因。

## 连接中断处理

第一波SSH stdout连接中断后，原独立guard和训练进程继续运行。三个方向修正头与运动预测头最终均exit0，末头156轮双平台停止。只通过 `reconcile_completed_wave.py` 校验原四份guard退出、COMPLETED与权重hash后补齐缺失的整批回执，记录 `TRANSPORT_RECOVERY.json`；该操作启动0个训练进程，没有重试。

第二波及评估入口仅将已授权调度器独立于SSH运行，输出落盘到 `transport_launches`；每个计算子任务原有30秒心跳、硬限、无自动重试语义不变。这不是扩实验范围或换方法。
