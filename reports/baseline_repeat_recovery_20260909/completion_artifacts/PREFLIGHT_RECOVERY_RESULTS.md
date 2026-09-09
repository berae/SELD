# 两份baseline预检修正：通过

## Material Passport

- 2026-09-09；academic-research-suite / experiment-agent，授权执行。
- 两份各一次CPU预检，exit0，guard均约30.03秒，无优化器更新，无超时。此前失败记录保留在STATUS.md及旧guard中。
- 本文件仅确认预检，不预填后续训练或evaluation结果。

## 静态与运动信号分别核验

同一首条train录音的前两个chunk均有61个匹配、59个有效位移pair，位移全为0。独立运动预测头零初始化下，损失和所有梯度均严格为0且有限，符合预期。

同一录音全部15个chunk含324个非零目标坐标；seed2027有470个匹配、452个有效pair，seed2028为471、453。四头分别独立初始化，预检随机种子2026；末层梯度绝对值之和如下，全部有限且大于0：

| 简单名称 | baseline2027 | baseline2028 |
|---|---:|---:|
| 原模型＋方向修正 | 0.489917 | 0.818746 |
| 方向修正＋历史输入 | 0.482627 | 0.689655 |
| 方向修正＋历史输入＋运动监督 | 0.501602 | 0.696402 |
| 原模型＋独立运动预测头 | 0.239495 | 0.225025 |

三个方向修正头的零输出恢复原始浮点和CSV；所测非零头前缀一致。原输出、两帧平滑、卡尔曼的全部validation评分和逐录音CSV复现上次预检已产出的固定对照，纯SED计数一致；完整缓存hash、baseline身份和train/validation不交叉检查通过。

每个chunk的历史起点重置保持不变。预检没有修改头参数值，没有加载或更新主干，已封存代码快照前后hash不变。训练数据、网络、损失、训练/停止/选优策略未改。

## 来源与传输

- [授权](PREFLIGHT_RECOVERY_AUTHORIZATION.md)。[baseline2027完整回执](preflight_recovered/C0_2027/COMPLETED.json)，[baseline2028完整回执](preflight_recovered/C0_2028/COMPLETED.json)。
- 实际命令、PID、日志、退出：`artifacts/guards/preflight_recovered_2027`、`artifacts/guards/preflight_recovered_2028`。
- 预检入口SHA256：`b3992fb5d74e205297573893de59da9710a85dcf328b5c132d4d29f6a8793dc9`，两端一致。
- 整包SHA256：`242e3bd048651361ad4113fceeeff7fe8e71667315a57f46e32fd54cc0bd5fa0`，两端一致。首次逐文件传输中断，只留下141个文件；后从校验通过的整包补齐483个缺失文件，已有文件逐个核验相同，没有覆盖旧实验记录。
- 预检600份CSV为validation固定对照的回归副本，不是新增evaluation条件或新训练次数。

已解除静态样本误触非零梯度断言造成的暂停；后续仍须八头全部冻结后再evaluation。按技能保留了失败、修正授权及正负预检证据，未将门槛改成容差放行。
