# 预检取样修正授权

## Material Passport

- 2026-09-09；academic-research-suite / experiment-agent，run。
- 用户针对上一轮明确问题回复“允许”：仅修正预检取样，同一首条训练录音全部15个chunk用于非零梯度测试；前两个静态chunk保留为零目标零梯度检查。固定预检随机种子2026。
- 两份baseline各一次无优化器CPU预检，每份1800秒硬限，新目录 `preflight_recovered/C0_2027`、`preflight_recovered/C0_2028`；保留旧失败日志与固定对照，不重建缓存。
- 新入口 `operations/preflight_recovered_baselines.py`，不改已封存 `code/` 快照，模型／损失／训练／选优核心不变。成功后只创建此前不存在的各baseline `PREFLIGHT.json`。
- 命令：既有Python与guard调用新入口，参数 `--root /work/zhanghc/Myllm/SELD/reports/baseline_repeat_recovery_20260909 --C0-seed <2027或2028> --output <根>/preflight_recovered/C0_<seed>`。实际命令/PID/时间由 `guards/preflight_recovered_<seed>` 保存。
- 两份预检均通过后沿用原8头授权，顺序两波、每波最多4个独立head seed2026；只用rabbit02空闲3090，每头14400秒，min40/max200双平台。全部8头冻结后再新evaluation。
- 不改变门槛、不自动重试、不扩方法/数据集、不重算既有bootstrap。历史evaluation不称盲测。
