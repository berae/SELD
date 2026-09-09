# 两份 baseline 缓存恢复与重复验证授权回执

## Material Passport

- 2026-09-09；academic-research-suite / experiment-agent，run。
- 唯一 v2；仅执行已登记两份 baseline 的重复验证，不新增方法。
- 本回执记录当前会话用户回复“允许”：同意最小导出修复，在新目录重建两份 baseline 的 train/validation 缓存；全部严格核验通过后继续原8头训练，全部冻结后统一评估。

## 范围与不变量

- rabbit02 空闲3090，每卡一任务，最多4头并行；RB05不计算。
- 仅 seed2027/2028 导出启用 `--preserve-source-model-flags`：模型保持历史默认参数标志，前端保持冻结，eval + torch.no_grad，无优化器/反向传播，参数及缓冲区值不变。参数标志不是训练授权。
- 新根目录 `/work/zhanghc/Myllm/SELD/reports/baseline_repeat_recovery_20260909`。旧目录、失败记录及所有历史参考保持原样。
- 每份 train500 / validation100 重新导出；不混用旧train。validation全部浮点逐位、CSV字节、全量评分严格回归；比较新旧train特征及字段差异留档。
- 缓存每任务1800秒；此前60分钟仅适用于已授权诊断，不扩到缓存。每头14400秒，min40/max200双平台、validation-best、独立head seed2026，最多8头。模型、损失、停止/选优核心不变。
- 两份baseline各四头全部封存后才做新evaluation；历史evaluation已查看，不称盲测。不挑seed、不重算既有bootstrap。
- 不自动重试失败任务，不放宽任何核验门槛。

## 登记的执行入口

所有入口从新根目录 `code/refinement_v2` 的经hash核验快照执行；Python沿用 `/work/zhanghc/Myllm/SELD/EINV2/.venvs/einv2-cu121/bin/python`。

1. `prepare_cross_c0.py --root <新根> --previous /work/zhanghc/Myllm/SELD/reports/refinement_v2_confirmation_20260908_r1 --preserve-source-model-flags`。
2. `launch_cross_c0.py --root <新根> --stage cache`，四个缓存任务，逐卡复查空闲；实际命令、GPU UUID和PID由dispatch/guard登记。
3. 新旧train一致性审计，随后分别 `preflight_cross_c0.py --root <新根> --C0-seed 2027` / `2028`，无优化器预检。
4. 分别 `launch_cross_c0.py --root <新根> --stage train --C0-seed 2027` / `2028`，顺序两波。
5. `freeze_cross_c0.py --root <新根>`，核对 `ALL_EIGHT_FROZEN.json` 后，按各自封存配置导出evaluation200，再 `evaluate_frozen.py` 评估全部七条件。

具体命令和输出路径在各阶段启动前落盘；本文件不是结果通过证明。
