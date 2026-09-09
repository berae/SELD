# EINV2 0.2.0：当前帧因果重跑协议

用户于2026-09-04确认采用**当前100 ms帧**，不要求帧起点的瞬时判断。保留原STFT、时间池化和4 s chunk：标签帧j的最晚音频依赖为`j*100+75 ms`之前，不使用下一标签帧音频。称为 **frame-causal / current-frame causal**，不称 waveform-level zero-look-ahead。75 ms不是实测系统总延迟；当前没有流式缓存或实时部署验证。

## 固定矩阵与配置

| Variant | Velocity λ | Latent λ | Seeds |
|---|---:|---:|---|
| C0 | 0 | 0 | 2026,2027,2028 |
| C1 | .2 | 0 | 2026,2027,2028 |
| C2 | 0 | .2 | 2026,2027,2028 |
| C3 | .2 | .2 | 2026,2027,2028 |

- TAU2020 FOA 24 kHz，14 classes；train folds2–6，validation fold1，独立evaluation200段。
- 4 logmel + 3 IV，256 mel，nfft1024，hop600；4 s不重叠chunks。
- 四组都实例化相同`AuditedEINV2`（继承冻结C3网络）。辅助heads均存在，关闭的loss分支不反传；公共权重、head初始化、dropout顺序与data sampler保持配对。
- 保留主干、velocity SmoothL1、JEPA cosine、horizons[1,3,5]和EMA=.996；teacher依据自身SED/DOA独立tPIT匹配再提供future-latent目标。标签只在训练loss中使用，不进入推理输入。
- 90 epochs，batch32，Adam(lr=.0005, amsgrad=True)，第80个完整epoch后StepLR×.1；beta=.5，无新增augmentation。
- PyTorch deterministic algorithms开启、cudnn benchmark关闭，固定CUBLAS配置；不据此声称已跨环境复现。
- scaler仅用folds2–6，通道顺序为`logmel_WYZX_IV_XYZ`；训练/validation/独立inference使用同一个`Frontend`和相同batch size。
- 每epoch按DCASE2023官方core、micro、20°、完整60 s/1 s blocks计算validation；`best.pth`只按最小`SELD_LR`选择。evaluation不参与选优。`F20`不是纯SED F1，未新增mAP。

## 版本边界

`models/einv2/variants/*`、旧checkpoint和报告保持不变。0.2.0是新实验系列，不直接与旧表拼接平均。[历史审计问题](../reports/EINV2_REAUDIT_20260904.md)尚不能解释历史效果大小。本轮优先causal C0–C3；offline E0–E3不在这12个run内。旧架构图描述冻结实现，新runtime的teacher独立对齐和共享初始化以本文为准。

## 入口（从仓库根目录执行）

```bash
CUDA_VISIBLE_DEVICES=4 python scripts/data/build_einv2_train_scalar.py \
  --project-root /work/zhanghc/Myllm/SELD \
  --output /work/zhanghc/Myllm/SELD/experiment_einv2_reaudit_20260904_r1/scalar_trainfolds2-6.h5

CUDA_VISIBLE_DEVICES=0 python scripts/train/einv2_audited.py \
  --project-root /work/zhanghc/Myllm/SELD \
  --output-root /work/zhanghc/Myllm/SELD/experiment_einv2_reaudit_20260904_r1/runs \
  --scalar /work/zhanghc/Myllm/SELD/experiment_einv2_reaudit_20260904_r1/scalar_trainfolds2-6.h5 \
  --variant C0 --seed 2026

CUDA_VISIBLE_DEVICES=0 python scripts/eval/einv2_audited.py \
  --run /work/zhanghc/Myllm/SELD/experiment_einv2_reaudit_20260904_r1/runs/C0_seed2026_framecausal_v020 \
  --split evaluation
```

路径保存在每个完整run config中。上述scaler/训练命令遇到已存在产物会退出，不覆盖历史结果。

`scripts/train/einv2_audited_queue.py`提交固定12-run队列，显式传`--gpus`；训练成功后分别调用validation/evaluation入口。每30秒记录进程状态，10分钟无日志仅提示，单阶段默认hard timeout24小时。失败不重试并停止派发新的训练，已经运行的其他作业保留。不会驱逐其他GPU用户。

## 输出与验证

每run保存`config.json`（含源码/scaler SHA256）、`environment.json`、`split_manifest.json`、`status.json`、`metrics.jsonl`、`best.pth`、`latest.pth`。checkpoint包括model、frontend、teacher、optimizer、scheduler和随机状态。目前仅支持fresh run，未声称支持精确断点续训。

独立validation核对训练时best-checkpoint metric；写出CSV后再读入重评分，必须一致。独立evaluation不创建optimizer/teacher。主训练阶段不读取evaluation数据。

已完成7项CPU回归及C3两batch GPU smoke +100段validation/独立validation，后两者metric最大差0。Smoke预测为空，只证明执行/产物链可用，不作为性能证据；真实训练后非空预测的独立一致性仍需后续检查。

运行`python tests/test_einv2_audited.py`执行回归测试。本轮证据在`reports/reaudit_20260904/`；最新进度以rabbit02的`queue_status.json`为准，仓库快照不是实时监控。
