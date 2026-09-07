# 唯一 v2 执行代码

本目录围绕同一个冻结 C0 的检测保持方向修正，不是额外主干或新数据实验。

## 配置与授权变更

- `configs/refinement_v2/pilot.json` 是首批交付原件，保留其历史状态与哈希。
- `configs/refinement_v2/execution_r1.json` 是本次授权追加：用户允许补齐代码、在空闲卡运行并优先 3090；随后要求优先收敛，不以 20 epochs 结束。
- 本轮四个学习条件统一至少 40、最多 200 epochs。连续 20 epochs 验证 SELD_LR 未改善超过 1e-5，且训练损失前后两个十 epoch 窗口均值相对变化不超过 1%，才以双平台规则提前停止。每个条件总进程硬限 4 小时，达到 epoch/时间上限不等于确认收敛。
- 验证集每 epoch 打分；严格最小分选优，并列取早。提前停止进展门槛与最优模型选择分开。无 scheduler、无 sweep，仍使用原 lr=.001/AdamW/batch32/hidden128/seed2026。
- R0/R1/R2 是递进消融，F-Deriv 重新在同 C0 特征上拟合。没有新增 2×2 条件、JEPA、主干训练或 evaluation 选参。
- 单 seed 的双平台仅是操作性收敛判据，不保证最优解，也不支持确证性优越结论。
- 最新资源边界：用户要求暂不使用 RB05 计算，仅允许只读复制固定 C0 的源文件到 rabbit02。使用 `relocation_rabbit02_r1.json` 只替换存储路径；权重/scaler/runtime 哈希及原 validation 回归门槛不变。不能把 rabbit02 原有另一批 C0 当成同一个 C0。

## 入口

- `cache_features.py`：原环境原 batch32/no_grad 原 forward 导出，保存 h、logits、概率、raw xyz、记录 ID、预测关联；GT 匹配和 motion mask 单独保存为 targets；核对全部 validation 原始浮点、CSV 字节、官方分数及首末批波形前缀。
- `experiment_core.py`：固定目标、预测关联上的固定 Kalman、批量 head 与 loss；GT 不用于输入关联。
- `test_execution.py`：新增执行接口的合成 CPU 回归，包含梯度、空 mask、GT 身份不一致、换槽与跨 chunk。
- `train_heads.py`：F0/F-EMA/F-KF 完整 validation；或 F-Deriv/R0/R1/R2 的独立 smoke/收敛试跑。只加载冻结缓存与新 head，不加载主干。
- `run_guarded.py`：独立进程、30 秒心跳、日志停滞提示、硬时限；失败不自动重试。
- `diagnose_fixed_targets.py`：七个条件完成后，对原 C0 固定匹配目标做量化方向误差的描述性分层；不重新匹配，不代替官方 LE_CD，不输出显著性结论。

已完成执行与证据入口：[单 seed 收敛试跑结果](../../reports/refinement_v2_execution_20260907_r1/RESULTS.md)。

## 调用规范

实际路径和 GPU 分配记录在每个新运行目录的 `RUNNING.json` / `STARTED.json`，不把示例视为已执行。

```text
python cache_features.py --config <pilot.json> --output <new-cache-split> --split validation
python cache_features.py --config <pilot.json> --output <new-cache-split> --split train
python test_execution.py --output <new-test-report.json>
python train_heads.py --cache <cache-root> --config <pilot.json> --execution <execution_r1.json> --scorer <pinned-evaluation-directory> --output <new-run> --condition R0 --smoke-batches 3
python train_heads.py --cache <cache-root> --config <pilot.json> --execution <execution_r1.json> --scorer <pinned-evaluation-directory> --output <new-run> --condition R0
```

所有输出目录必须不存在。smoke 权重不延续到正式试跑。完整文件清单和哈希验证通过后才能训练。移动缓存不改变 C0/数据条件，所有源 checkpoint、runtime 和标签哈希必须一致。

## 证据与限制

- 训练缓存是 C0 的 in-sample 表示，须披露训练/验证难度差异，不自动交叉拟合。
- 只改变已有检测记录方向；FP 保留，FN 不能恢复；概率及纯 SED 不作为新增优势。
- 静态/运动 pair 阈值与 unmatched 正则的可执行定义，在 execution 配置中先于训练结果明确。
- 每 epoch 的损失、有效分母和全部评分保留。最优权重及验证 CSV/逐录音指标保留，但不保证所有条件收敛或改善。
- 原 `candidate.py`、原 `pilot.json` 与首批证据均不覆盖。大量数据、缓存、权重、凭据不得提交到 Git 仓库。
