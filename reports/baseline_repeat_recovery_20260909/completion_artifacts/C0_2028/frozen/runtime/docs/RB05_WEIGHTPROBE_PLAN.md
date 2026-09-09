## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-04
- Verification Status: UNVERIFIED
- Version Label: einv2_rb05_weightprobe_v1

## 第一阶段：JEPA 权重单变量验证

用户授权在 RB05 逐步验证。仅使用 GPU 0、1；不改变 rabbit02 实验。
父版本：repository 0.2.0，commit 5608354aa0035472bfd811673ebce538e7fd17e7。

| Run | lambda_velocity | lambda_jepa | Seed | Epochs |
|---|---:|---:|---:|---:|
| C3_jepa020_seed2026_rb05_v1 | 0.2 | 0.2 | 2026 | 90 |
| C3_jepa005_seed2026_rb05_v1 | 0.2 | 0.05 | 2026 | 90 |

两组均从头训练，共用初始化、训练样本顺序、dropout 随机流、batch 32、Adam
5e-4、epoch 80 后学习率乘 0.1、EMA 0.996、horizons [1,3,5]。
保留当前帧因果定义、4 s chunk、velocity head 和 predictor；不增加 warm-up，
不修改 mask、loss 公式或 backbone。0.05 是待检验候选，不宣称是最优值。

训练使用 TAU2020 FOA folds 2–6（500 files）；验证 fold 1（100 files）。
使用从 rabbit02 复制且校验一致的 train-only scalar（7,256）。
RB05 HDF5 指向 EINV2/shared/_hdf5，不复制大型历史 checkpoint 或原始数据。
RB05 本机同硬件对照，避免直接把跨机器差异归因于 loss。

## 验证顺序与边界

1. 检查数据覆盖、形状、标签和抽样音频 hash；运行 7 项既有回归和 2 项权重回归。
2. 两组分别做 2 batch smoke + 全部 100 validation files + 独立 CSV 重评估。
   smoke 仅验证执行链，不是性能证据；smoke 的空预测不证明非空预测数值一致。
3. 通过后两卡各跑一个完整 90 epoch run；根据同一 validation SELD_LR 选 best。
   保留每个 epoch 的 LE/LR/F20/SELD_LR 和 loss。
4. 训练完成后仅独立导出 validation，测试集保持封存，不做测试集调参。
5. 在验证结果上比较总体和静态/动态 LE、配套 LR/匹配召回、纯 SED F1/mAP，
   以及共同匹配目标的定位误差。F20 不能代替纯 SED F1；后续诊断另行执行。
   单 seed 只能筛选候选；不能据此声称显著改善或互补性成立。

后续阶段（本轮不自动启动）：单独验证 warm-up；单独验证直接 DOA 差分监督；
连续性/常量退化对照；有依据后再扩展三 seed 与正式 test evaluation。

## 运行监控

有限两任务队列，training 和独立 validation 各设 24 h hard timeout。
每 30 s 检查存活和输出；10 min 无输出仅标记 advisory。
失败不自动重试；开始前检查 GPU 空闲，不占用其他人的任务。
source hash、配置、环境、split manifest、checkpoint 和日志保存在独立实验目录。
