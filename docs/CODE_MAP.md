# 代码地图与运行入口（0.3.0）

本轮整理不改动模型数学实现，不移动服务器数据或权重，不启动训练。新增显式recipe入口、恢复旧runtime的工具、结果总账，并收录此前未入库的历史项目代码。

## 当前实现

| 内容 | 入口/目录 | 用途 |
|---|---|---|
| EINV2 当前帧因果 | `models/einv2/audited/` | 与RB05 dynamicmask冻结版本逐字节一致；默认norm=0保留all-valid-pairs行为 |
| EINV2 C0–C3 / E0–E3历史模型 | `models/einv2/variants/` | 独立变体，不贸然合并；E0调用C0目录下offline EINV2类 |
| Multi-ACCDOA | `models/multi_accdoa/` | 与EINV2独立；historical/便携recipe分开 |
| 训练：新版EINV2 | `scripts/train/einv2_recipe.py` | 一次一个recipe/seed，仅训练与validation；显式指定host路径 |
| 训练：原始入口 | `scripts/train/einv2_audited.py` | 保留原参数；原run必须使用匹配runtime |
| 训练：D1/D3队列 | `scripts/train/einv2_dynamicmask_queue.py` | 正确命名的新别名，调用冻结队列，不自动执行 |
| 训练：历史模型 | `scripts/train/einv2.py`、`scripts/train/multi_accdoa.py` | 对应legacy配置，不代表修复后EINV2 |
| 独立推理/测试 | `scripts/eval/einv2_audited.py`、`einv2.py`、`multi_accdoa.py` | 不启动optimizer/训练 |
| 官方评分/分层 | `scripts/eval/score.py`、`motion_strata.py` | DCASE版本与自定义motion诊断分开 |
| 预处理 | `scripts/preprocess/`、`scripts/data/` | features、velocity labels、JEPA masks、train-only scalar |
| 配置 | `configs/einv2/audited/*.json` | C0/C1/C2/C3、低JEPA、D1/D3；planned文件明确标记未完成实验 |
| 机器路径 | `configs/hosts/*.example.json` | 实际核查的rabbit02/RB05项目与HDF5/scalar路径；不存凭据 |
| 结果生成 | `scripts/reports/build_all_results.py` | 仅读取保存证据，重算表格；不访问服务器/GPU |
| 精确旧runtime恢复 | `scripts/reproduce/materialize_einv2_release.py` | 根据训练source hash恢复到新目录 |

## 新训练命令示例（本轮未执行）

先检查计划；GPU编号由使用者指定，不自动占卡。

```bash
python scripts/train/einv2_recipe.py \
  --recipe configs/einv2/audited/C3_jepa005.json \
  --paths configs/hosts/RB05.example.json --seed 2026 \
  --output-root /home/zhanghc/SELD/experiments/new_run/runs --dry-run
```

确认后去掉`--dry-run`，指定`CUDA_VISIBLE_DEVICES`再训练。该入口沿用冻结runtime，不把整理后的结果称为已重训。配置最终展开后保存在新run的`config.json`，运行名带`v030`；算法runtime仍为`einv2_rb05_dynamicmask_v1`。

原 `einv2_weightprobe_queue.py` 在dynamicmask快照内实际是D1/D3队列，因它被checkpoint源码hash约束而不能直接重命名；不要用该旧文件名猜任务。真正旧weightprobe队列可按以下方式恢复。

## 已有checkpoint的正确复现

| config.json 中 `audit.version` | 所对应训练 |
|---|---|
| `einv2_reaudit_v1` | rabbit02 C0–C3三seed |
| `einv2_rb05_weightprobe_v1` | RB05 C3 JEPA .05 / .2 |
| `einv2_rb05_dynamicmask_v1` | RB05 B0/B1 fairbaseline、D1/D3 |

```bash
python scripts/reproduce/materialize_einv2_release.py \
  --version einv2_reaudit_v1 --output /tmp/seld_reaudit_restore
```

恢复工具仅复制source manifest中的文件，不复制数据/checkpoint。随后在原服务器用该恢复目录下的` scripts/eval/einv2_audited.py --run /absolute/run --split evaluation`。原评价目录已存在时入口会拒绝覆盖；已有结果直接读或重评分，不必重复推理。

checkpoint中绝对数据路径保留原样，跨机器重放仍需明确迁移/路径处理，恢复source并不自动改写checkpoint。不要为绕过校验修改hash或去掉检查。

## 历史代码归档

`archive/legacy_projects/rabbit02/` 新收录168份原字节源文件/配置/说明：

- `ObjectStateSELD/`：工程B0、数据manifest与smoke工具；B1–B3未完成，非当前方法。
- `EINV2/STARSS_OfficialEINV2/`：STARSS历史训练实现/配置；结果与TAU2020分开。
- `MultiACCDOA_TAU2020/`、`CausalMultiACCDOA_TAU2020/`、`MultiACCDOA_DCASE2023/`：早期baseline脚本/配置，仅历史复现入口。

归档目录可能含旧硬编码路径及一体化train/eval脚本，这是原始来源，不作为推荐入口；当前入口仍按train/eval分开。不声称归档代码全部可独立安装运行。原始上游许可与归属继续有效，不重新授权。

`PSELDNets`、`NTU_SNTL_Task3`、ObjectState的`external/`等第三方目录不整库复制；服务器索引记录Git/路径。数据、缓存、环境、weights、完整prediction不入Git。源码整理范围和原始路径见`provenance/export_*_20260907.json`与旧`source_snapshot.json`。

## 结果入口

- [全实验报告](../reports/ALL_EXPERIMENTS.md)：68个统一主run，另附旧8×3 Multi矩阵、STARSS曲线、全部旧记录。
- [逐run总账](../reports/summary_20260907/runs.csv)、[旧manifest原始数值](../reports/summary_20260907/original_multi_results.csv)。
- `reports/summary_20260907/evidence.json`是原始材料索引，`raw/`存逐条JSON、SHA256、配置、评分、状态等；没有凭空补零的缺测值。

复算：`python scripts/reports/build_all_results.py`。源码与结果分开提交，服务器冻结目录保持原状。
