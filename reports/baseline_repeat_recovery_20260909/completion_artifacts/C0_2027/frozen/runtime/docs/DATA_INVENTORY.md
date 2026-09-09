# 数据资产清单

核查日期：2026-09-04。来源是 rabbit02 / RB05 的实际文件，不是根据目录名称或聊天记录推测。**当前主实验使用 TAU2020；已有六套原始数据，但“数据存在”不等于“本仓库已接入训练”或“实验完成”。** 本次只读扫描，未训练、未修改原始数据；仓库只保存清单、检查脚本和元数据证据。

路径缩写：`R02 = /work/zhanghc/Myllm/SELD`；`R05 = /home/zhanghc/SELD`，后者通过 SSH alias `RB05` 访问，实际 hostname 为 `rabbit05`。检查范围为这两个项目目录及相关链接目标，不是两台服务器的全盘搜索。

## 1. 原始数据总表

容量为音频与 metadata 的逻辑文件大小，单位 GiB（2^30 bytes），不含下载压缩包；时长由本地音频 header 求和。FOA 为一阶 Ambisonics，MIC 为麦克风阵列格式。**下表所有音频均为 24 kHz / PCM_16；FOA、MIC 为 4 channels，Stereo 为 2 channels。**

| Dataset | 已有音频 / split | 有效标签 CSV | 音频时长 | 容量 GiB | rabbit02 原始路径 | RB05 | 当前可用范围 |
|---|---|---|---|---:|---|---|---|
| TAU-NIGENS 2020 | dev 600 + eval 200；每条 60s | dev 600 + eval 200 | 10.00h + 3.33h | 8.590 | `R02/ObjectStateSELD/data_raw/dcase2020` | 已有完整对应副本，见迁移表 | 当前 EINV2 / Multi-ACCDOA 主实验；train folds2–6、validation fold1、独立 evaluation 200 |
| TAU-NIGENS 2021 | dev train 400 / val 100 / test 100；eval 200；每条 60s | dev 600 + eval 200 | 10.00h + 3.33h | 8.593 | `R02/datasets/TAU2021/raw` | 本次项目范围内未发现 | 原始数据及标签齐备；未接入本仓库 v0.1.0 实验配置 |
| STARSS22 | dev train 67 / test 54；eval 52 | dev 121；未见 eval 标签 | dev 4.87h + eval 1.92h | 4.374 | `R02/datasets/STARSS22/raw` | 本次项目范围内未发现 | 独立 `STARSS_OfficialEINV2` 工程已有缓存；不属于本仓库当前 TAU2020 matrix |
| STARSS23 | dev train 90 / test 78；eval 79 | dev 168；未见 eval 标签 | dev 7.37h + eval 3.55h | 7.037 | `R02/datasets/STARSS23/raw` | 本次项目范围内未发现 | 独立 STARSS / PSELDNets / Multi-DCASE2023 目录引用；不表示已接入当前发布入口 |
| DCASE2024 Synthetic | FOA 1,200 + MIC 1,200；每条 60s | 1,200，同名场景共用 | 每种格式 20.00h；场景不翻倍 | 25.776 | `R02/datasets/DCASE2024_Synthetic/raw/DCASE Task 3 synthetic dataset 2024` | 本次项目范围内未发现 | 两种音频格式及对应标签齐备；未接入本仓库 v0.1.0 实验配置 |
| DCASE2025 StereoSELD | dev train 16,214 / test 13,786；eval 10,000；每条 5s | dev 30,000；未见 eval 标签 | dev 41.67h + eval 13.89h | 17.913 | `R02/datasets/DCASE2025_StereoSELD/raw` | 本次项目范围内未发现 | 原始 Stereo 数据已有；不能直接当作 FOA 输入现有模型 |

本轮读取了六套原始数据全部 44,420 个 WAV 的 header，未出现 header 读取错误。**这不是 44,420 个独立场景**：DCASE2024 的 FOA/MIC 对应相同场景；STARSS22/23 也不能假定彼此独立，未做跨数据集全内容去重。不能把各行时长直接相加当作独立训练数据量。

带标签的 11 组 audio/metadata 配对检查全部通过，录音相对路径名无缺失、无多余标签；包括 RB05 的 TAU2020 dev/eval。STARSS22/23、DCASE2025 的 eval 音频存在，但在所检目录未发现 `metadata_eval`，因此不能据此宣称可本地完成有标签 evaluation。配对检查不等于逐行验证标签语义。证据：[label_coverage.json](data_inventory/2026-09-04/label_coverage.json)。

## 2. 派生特征、标签与训练缓存

以下容量不能再当作新增数据集容量或样本量；可能包含同一音频的不同表示。shape 为样本检查，不是全文件逐元素校验。

| Asset | 文件数 / GiB | 内容与样本 shape | rabbit02 路径 | RB05 路径 / 状态 |
|---|---|---|---|---|
| Multi strict-causal features | 2,202 / 16.923 | dev raw 600 + norm 600 + ADPIT 600；eval raw 200 + norm 200；另 2 辅助文件。feature `(3000,448)`，label `(600,6,4,14)` | `R02/MultiACCDOA_TAU2020/features/tau2020_foa_multiaccdoa_strictcausal_trainfolds2-6` | `R05/DynamicCausalMultiACCDOA_TAU2020/features/tau2020_foa_multiaccdoa_strictcausal_trainfolds2-6` |
| EINV2 TAU2020 HDF5 | 1,606 / 8.647 | audio 800 + label 800 + scalar 2 + statistics 4；waveform `(4,1440000)`，SED `(600,2,14)`，DOA `(600,2,3)` | `R02/EINV2/C0_CausalEINV2_seed2026/_hdf5` | `R05/EINV2/shared/_hdf5` |
| EINV2 velocity targets | 600 / 0.014 | dev velocity `(600,2,3)` + mask `(600,2)` | `R02/EINV2/C1_CausalEINV2_Velocity_seed2026/velocity_hdf5` | `R05/EINV2/C1_CausalEINV2_Velocity_seed2026/velocity_hdf5` |
| EINV2 JEPA targets/masks | 600 / 0.017 | dev identity `(600,2,2)` + mask `(600,2,3)`；不是额外录音 | `R02/EINV2/C2_CausalEINV2_JEPA_seed2026/jepa_hdf5` | `R05/EINV2/C2_CausalEINV2_JEPA_seed2026/jepa_hdf5` |
| Multi legacy features | 2,201 / 16.923 | 历史前端/normalization 缓存，不能与 strict 版本混用 | `R02/MultiACCDOA_TAU2020/features/tau2020_foa_multiaccdoa` | 本次未发现 |
| EINV2 STARSS22/23 HDF5 | 1,016 / 11.489 | waveform HDF5 420；track-label HDF5 289；另有 scalar、frame CSV 等 | `R02/EINV2/STARSS_OfficialEINV2/_hdf5` | 本次未发现 |
| PSELDNets HDF5 | 5 / 0.126 | 3 个 label-container HDF5 + 2 CSV；所检目录未见 audio HDF5，不是完整 waveform 缓存 | `R02/PSELDNets/_hdf5` | 本次未发现 |
| ObjectState intermediate | 9 / 0.053 | parsed tracks / manifest / NPZ / JSON 等中间产物 | `R02/ObjectStateSELD/data_intermediate` | 本次未发现 |
| ObjectState processed | 0 / 0 | `v0_single_track_2s_05s` 为空，不能记为已完成 processed dataset | `R02/ObjectStateSELD/data_processed` | 本次未发现 |

EINV2 `--hdf5-root` 指向表中的 `_hdf5`；`--velocity-root` 应指向 `velocity_hdf5/dcase2020task3/velocity`，`--jepa-root` 指向 `jepa_hdf5/dcase2020task3/jepa`，后二者直接包含 `dev/`。两个 scalar 均已迁移，但 **causal / offline 的 IV 通道顺序与 scalar 约定必须匹配**，不能因 shape 相同就互换。详见 [实验协议](EXPERIMENT_PROTOCOL.md)。

## 3. rabbit02 → RB05 迁移核对

TAU2020 的 RB05 原始音频和标签位于 `R05/DynamicCausalMultiACCDOA_TAU2020/data/TAU2020_SELD_dataset`，各 split 使用 `foa_dev/source`、`foa_eval/source`、`metadata_dev/source`、`metadata_eval/source`。以下均为**本轮重新核对**，不是把历史迁移退出码当作完整性证明。

| 对应资产 | 文件数 | 相对文件名 + 文件大小全量比对 | 内容 SHA256 抽样 | 全内容检查范围 | 结论 |
|---|---:|---|---:|---|---|
| TAU2020 Multi data view | 1,600 | 一致 | 12 / 12 一致 | dev/eval 共 800 个有效标签全部一致 | 文件清单、大小、抽样及标签一致 |
| Multi strict features | 2,202 | 一致 | 15 / 15 一致 | 未全量 hash 特征 | 文件清单、大小、抽样一致 |
| EINV2 TAU2020 HDF5 | 1,606 | 一致 | 14 / 14 一致 | 2 个 scalar 全部一致 | 文件清单、大小、抽样及 scalar 一致 |
| EINV2 velocity targets | 600 | 一致 | 3 / 3 一致 | 未全量 hash targets | 文件清单、大小、抽样一致 |
| EINV2 JEPA targets/masks | 600 | 一致 | 3 / 3 一致 | 未全量 hash targets | 文件清单、大小、抽样一致 |

合计 6,608 个文件，47 个确定性抽样全部一致；另对 800 个标签和 2 个 scalar 做了全内容核对（与抽样可能重叠）。**未对所有音频、特征和 HDF5 做全量内容 checksum，不能声称全量 bitwise 完整性已验证，也不代表新训练已验证成功。** 证据：[migration_comparison.json](data_inventory/2026-09-04/migration_comparison.json)。

## 4. 别名、压缩包与已知缺口

| 情况 | 实际含义 / 使用注意 |
|---|---|
| rabbit02 `EINV2/dataset_root` | 指向 `ObjectStateSELD/data_raw/dcase2020`，不是额外数据 |
| rabbit02 Multi / Dynamic / Causal 的 data、features，以及 EINV2 各变体缓存 | 多处软链接共用原始数据或 C0/C1/C2 缓存；不能按实验目录数量重复计数 |
| rabbit02 `MultiACCDOA_TAU2020/data/TAU2020_SELD_dataset` | 有 802 个目录/文件软链接；RB05 对应目录是 1,600 个实际文件，不是悬空的远程链接 |
| RB05 `EINV2/dataset_root` | metadata view；音频在 Multi dataset，EINV2 waveform 缓存在 `EINV2/shared/_hdf5`，并非原始音频未迁移 |
| `PSELDNets/datasets/STARSS23`、`MultiACCDOA_DCASE2023/data` | 引用已有 STARSS23，不另算新数据集 |
| `R02/ObjectStateSELD/data_raw/dcase2021` | 旧目录仅有 metadata / 文档等，未见 WAV；完整 TAU2021 在 `R02/datasets/TAU2021/raw` |
| `metadata_dev/._fold1_room1_mix001_ov1.csv` | rabbit02 TAU2020 raw 与 RB05 EINV2 metadata 中各有一个 AppleDouble sidecar；不是第 601 个标签。盘点按非隐藏 CSV 配对，未删除文件 |
| `R02/ObjectStateSELD/data_raw/dcase2020/archives/foa_dev.z01.corrupt-md5-1d208ecf` | 2 GiB、名称标记为历史损坏的分卷；不能作为有效备份。此次未重新验证该包内容，也未删除；不等于解压后的当前 WAV 有损坏 |

在本次盘点资产中未发现 broken symlink；不是对项目之外路径的保证。下载目录见下，包含压缩包和少量校验/说明文件，**不再计入原始数据量**。

| 下载目录（rabbit02） | 文件数 | GiB | 已读到的历史校验记录（非本轮重算） |
|---|---:|---:|---|
| `R02/ObjectStateSELD/data_raw/p1_downloads` | 24 | 31.478 | 含 DCASE2024 五个分卷等；2024 audit 记录 source MD5 / ZIP CRC PASS |
| `R02/datasets/TAU2021/downloads` | 9 | 5.694 | `extraction_audit_20260901.txt` 记录 ZIP CRC PASS |
| `R02/datasets/STARSS22/downloads` | 4 | 2.830 | 本次未重算 archive checksum |
| `R02/datasets/STARSS23/downloads` | 4 | 4.704 | 本次未重算 archive checksum |
| `R02/datasets/DCASE2025_StereoSELD/downloads` | 3 | 12.626 | `download_audit.txt` 记录 v1.1.0 / Zenodo 15559774、MD5 / ZIP CRC PASS |

## 5. Evidence Index 与复核

| 证据 | 内容 |
|---|---|
| [rabbit02.json](data_inventory/2026-09-04/rabbit02.json) | 25 个资产路径、文件计数/大小、全量 audio header 汇总、样本 shape / SHA256、软链接情况；扫描开始于 2026-09-04 09:11 UTC |
| [RB05.json](data_inventory/2026-09-04/RB05.json) | 7 个候选资产路径（含不存在的 `datasets`）；同类检查 |
| [label_coverage.json](data_inventory/2026-09-04/label_coverage.json) | 11 组录音名与标签名配对结果、排除的 sidecar |
| [migration_comparison.json](data_inventory/2026-09-04/migration_comparison.json) | 五类主实验数据/缓存的跨主机对比，保留两端绝对路径 |
| [collect_inventory.py](../scripts/data/collect_inventory.py) | 只读扫描；全量 stat/audio headers，目录首/中/末文件确定性抽样；样本上限 32 MiB |
| [check_label_coverage.py](../scripts/data/check_label_coverage.py) | 按相对路径配对 WAV/CSV，排除隐藏 sidecar |
| [复核 notebook](../notebooks/data_inventory_20260904.ipynb) / [tests](../tests/test_data_inventory.py) | 不访问原始数据即可复核快照、迁移比对和配对逻辑 |
| `R02/datasets/TAU2021/extraction_audit_20260901.txt` | 历史解压审计 |
| `R02/datasets/DCASE2024_Synthetic/extraction_audit_20260901.txt` | 历史 source MD5 / CRC 与 2,400 WAV、1,200 CSV 记录 |
| `R02/datasets/DCASE2025_StereoSELD/download_audit.txt` | 历史下载校验记录 |

服务器端重新扫描（需 `soundfile`；样本 shape 检查另需 `numpy` / `h5py`）：

```bash
python scripts/data/collect_inventory.py --host rabbit02 --root /work/zhanghc/Myllm/SELD
python scripts/data/collect_inventory.py --host RB05 --root /home/zhanghc/SELD
python scripts/data/check_label_coverage.py --audio /path/to/audio --labels /path/to/metadata
```

这些命令只输出结果，不写入数据目录。快照属于 2026-09-04 的资产状态；新增/迁移后应生成新的日期快照，保留旧记录。ARS `experiment-agent / validate`，Material status：`ANALYZED`；结论限于上述文件库存与验证粒度，不包含模型效果、标签语义完整性或全量大文件内容保证。
