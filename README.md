# SELD

EINV2 与 Multi-ACCDOA 的 causal / offline、velocity、JEPA 辅助监督实验代码。
当前版本 **0.3.0**：汇总两台服务器全部已发现实验记录，整理命名配置与训练/测试入口，保留三个精确runtime快照及历史代码；不包含数据、权重或大体积预测。

## 当前入口（2026-09-08）

- 2026-09-09：[前端保持冻结的首批次补核](reports/baseline_frontend_frozen_check_20260909/RESULTS.md)已通过；仅恢复主干历史参数标志即可逐位匹配两份baseline的首批原输出。[上一轮诊断](reports/baseline_repeat_diagnostic_20260909/RESULTS.md)保留。全量缓存修复/恢复仍待确认，新增小头0/8。

- [当前全部实验总览与最新结果](reports/CURRENT_EXPERIMENT_SUMMARY_20260908.md)：历史68个主干run、唯一v2的12个小头及15条件evaluation分开汇总；跨C0新增训练0/8，失败记录完整保留。最新证据目录包含3,000份轻量evaluation预测CSV，但不含权重、特征或音频。下方9月7日历史总表保留当时状态，后补结果与状态更正见本入口。

- [冻结 C0 检测保持方向修正：唯一 v2](scripts/refinement_v2/README.md)：缓存与回归、固定 Kalman、同 C0 导数头、R0/R1/R2 递进消融和收敛优先试跑入口；新增代码通过组件测试不等于已取得新训练结果。
- [全部实验结果](reports/ALL_EXPERIMENTS.md)：68个统一主run、131个已评分run×split；另列旧Multi 8×3矩阵、STARSS训练曲线、试跑与缺测。
- [代码地图与复现命令](docs/CODE_MAP.md)：EINV2/Multi-ACCDOA分开，训练/测试分开，host路径与method recipe分开。
- [逐run证据](reports/summary_20260907/runs.csv)、[全部旧记录](reports/summary_20260907/legacy_registry.csv)。旧A1/A2 λ=.2已有三seed，不能概括为“仅有代码”。

下面保留0.2.0及历史使用说明；已有checkpoint请先按`audit.version`恢复匹配runtime，不能直接用最新版覆盖旧运行源码。

## EINV2 新实验入口（0.2.0）

[当前帧因果重跑协议与命令](docs/EINV2_AUDITED_RUNS.md)：C0–C3 × 三 seeds，保留100 ms帧内75 ms输入依赖，统一归一化、train-only scaler、随机路径、teacher轨道对齐和官方validation选优。训练使用`scripts/train/einv2_audited.py`，独立测试使用`scripts/eval/einv2_audited.py`。

**下方原 EINV2 命令与既有结果表属于历史版本**，不能作为上述修复已验证有效的证据。历史预处理问题见[重审报告](reports/EINV2_REAUDIT_20260904.md)。Multi-ACCDOA本轮未修改。

## 目录

```text
models/
  einv2/audited/       0.2.0共享C0–C3运行逻辑，历史模型不变
  einv2/variants/      C0、C1、C2、C3、E1、E2、E3；C0_legacy 仅历史参考
  multi_accdoa/       独立的 Multi-ACCDOA 实现、参数与 analysis 工具
configs/
  einv2/              8 个可迁移配置；historical/ 保留原始配置
  multi_accdoa/       8 个显式 lambda005 配置；historical/ 为旧诊断配置
scripts/
  train/              只训练与 validation
  eval/               checkpoint 推理、官方评分、静态/动态诊断
  preprocess/         特征、velocity targets、JEPA masks
  data/               只读数据盘点与标签覆盖检查
evaluation/           SHA256 固定的 DCASE metric、适配与回归测试
tests/                入口、来源完整性、模型 forward/causal 测试
requirements/         EINV2 与 Multi-ACCDOA 分开的依赖
reports/              小型结果表与证据索引
provenance/           原始路径/hash、整理改动清单
docs/                 协议、历史结果、版本管理、上游说明
```

E0 使用 `models/einv2/variants/C0` 中的 offline `EINV2` 类；通过 `configs/einv2/E0.yaml` 选择，不使用 causal 模型。
EINV2 首版保留独立变体实现，避免合并代码时改变历史 checkpoint 对应的方法。不同模型用独立进程与环境运行。

## 数据资产

[数据清单与迁移核对](docs/DATA_INVENTORY.md)：2026-09-04 实查 rabbit02 / RB05，列出六套原始数据、split / 标签 / 时长 / 容量、派生缓存、绝对路径与验证范围。TAU2020 是当前主实验数据；其他数据已下载不等于当前训练入口已接入。仓库不上传原始音频、特征或权重。

## 环境与配置

- EINV2：Python 3.10，依赖见 `requirements/einv2.txt`。
- Multi-ACCDOA：Python 3.8，依赖见 `requirements/multi_accdoa.txt`。
- 仅重评分：`requirements/metrics.txt`。现有环境验证通过；未重新创建空白环境验证完整安装。
- 数据为 **TAU2020 FOA**：train folds2–6、validation fold1、独立 evaluation 200 recordings。
- 标准 seeds：2026、2027、2028。每个入口一次运行一个 seed；GPU 由 `CUDA_VISIBLE_DEVICES` 明确指定，不自动占卡。
- EINV2 causal 与 offline 的 scalar / IV 顺序不同，不能共用错误的预处理缓存；见 [协议](docs/EXPERIMENT_PROTOCOL.md)。
- 所有命令从仓库根目录执行。`--help` 查看参数；`--dry-run` 仅显示计划，不验证数据存在、不训练。

## 训练与测试分开

EINV2 C3 训练（路径替换为自己的路径）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=2026 python scripts/train/einv2.py \
  --variant C3 --seed 2026 --dataset-root /data/TAU2020_SELD_dataset \
  --hdf5-root /cache/einv2 --scalar-path /cache/einv2/matching_scalar.h5 \
  --velocity-root /cache/velocity --jepa-root /cache/jepa \
  --output-root /experiments/einv2
```

Multi-ACCDOA C3 训练（默认显式使用 lambda_velocity=lambda_jepa=0.05）：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=2026 python scripts/train/multi_accdoa.py \
  --variant C3 --seed 2026 --dataset-root /data/TAU2020_SELD_dataset \
  --feature-root /cache/multi_strictcausal --output-root /experiments
```

训练入口不自动访问 evaluation set。Multi-ACCDOA 写出包含完整 params/checkpoint 的 manifest；独立测试直接读取它：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/multi_accdoa.py \
  --manifest /experiments/multi_accdoa/C3_seed2026_v010/manifests/RUN.json \
  --split evaluation --output-dir /experiments/eval/C3_seed2026
```

迁移后可用 `--checkpoint`、`--dataset-root`、`--feature-root` 覆盖 manifest 的机器路径。测试不创建 optimizer，也不调用训练循环。

EINV2 先用显式 checkpoint 生成预测，再按相同官方协议评分：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval/einv2.py \
  --variant C3 --checkpoint /checkpoints/C3_epoch_best.pth --seed 2026 \
  --dataset-root /data/TAU2020_SELD_dataset --hdf5-root /cache/einv2 \
  --scalar-path /cache/einv2/matching_scalar.h5 \
  --split evaluation --output-root /experiments/einv2

python scripts/eval/score.py \
  --prediction-dir /experiments/einv2/out_infer/ein_seld/C3_seed2026_v010_evaluation/submissions \
  --reference-dir /data/TAU2020_SELD_dataset/metadata_eval \
  --prediction-schema polar4 --output /experiments/einv2/C3_metrics.json
```

Multi-ACCDOA 保存的预测使用 `cartesian7`；validation 参考目录改为 `metadata_dev`，并增加 `--reference-glob 'fold1*.csv'`。官方评分要求预测/参考文件集合完全匹配，包含空预测文件。

## 结果与 metric 版本

共同协议是 **DCASE2023 官方 core + micro + 完整60秒**，同时保留 macro 与 legacy2020 输出。这里的“2023”表示 metric 版本，不表示使用了 DCASE2023 数据集；micro 也不是官方默认 macro。

- 当前结果：[统一结果说明](reports/RESULTS.md)、[每个 run](reports/aligned_20260904/aligned_run_metrics.csv)、[三 seed 汇总](reports/aligned_20260904/aligned_aggregate_metrics.csv)。
- 完整列表：[14 组主结果与静态/动态定位表](reports/EXPERIMENT_CATALOG.md)、[42 个逐 seed run](reports/catalog_20260904/RUN_INDEX.md)；保留 config/checkpoint/result 路径与缺失实验说明。
- 历史 **15.28° → 11.94°**：[原始日志与限制](reports/HISTORICAL_LE.md)。它是 seed2026/fold1 validation，不能当作三 seed test 均值。
- 旧 `LR20` 实际是 localization F；新 `LR_CD` 才是 recall。`F20` 也不是纯 SED F1，未报告 mAP。
- 静态/动态工具的 `Recall@20` 是自定义逐帧诊断，不等于官方 `LR_CD`。
- 新训练的 validation metric 已切换；历史 checkpoint 的选择并未重做。新旧训练结果必须区分版本。

## 方法与架构图

[方法与架构图](docs/METHOD_AND_ARCHITECTURE.md)：2 张辅助监督方法图、4 张 causal / noncausal 模型架构图，覆盖全部 16 个 portable recipes。SVG 与生成源码一并保存，区分主预测路径、辅助 head 和 EMA target。

## 验证与开发

```bash
python tests/test_repository.py
python tests/test_data_inventory.py             # 无需原始数据
python tests/test_experiment_catalog.py          # 结果与架构图可复算检查
python evaluation/test_alignment.py
python tests/test_entrypoint_config.py           # EINV2 环境
CUDA_VISIBLE_DEVICES= python tests/test_einv2_variant.py --variant C3
PYTHONPATH=models/multi_accdoa python tests/multi_accdoa/test_dynamic_aux.py
PYTHONPATH=models/multi_accdoa python tests/multi_accdoa/test_experiment_protocol.py
```

见 [版本管理](docs/VERSIONING.md)、[验证记录](docs/VALIDATION.md) 和 [第三方来源](THIRD_PARTY_NOTICES.md)。
`docs/upstream/`、`configs/*/historical/` 是来源归档，可能包含旧机器路径或过时说明，不是当前运行指南。
