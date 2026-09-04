# EINV2 重审：当前帧因果口径已确认

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: code audit + run
- Origin Date: 2026-09-04
- Verification Status: ANALYZED（已运行代码诊断；未进行训练复现）
- Version Label: einv2_reaudit_20260904_r1

## 当前状态

用户希望重新检查EINV2，尤其causal模型，然后重跑。已在rabbit02执行CPU诊断，并于用户确认“当前帧”口径后保留75 ms时间对齐。新runtime已通过7项回归和GPU smoke，旧代码/权重/预测未覆盖；正式12-run以服务器队列状态为准，不能将smoke当作完成实验。详见[运行协议](../docs/EINV2_AUDITED_RUNS.md)。

- 实验根目录：`rabbit02:/work/zhanghc/Myllm/SELD`，不是当前本地 workspace；该总目录不是 Git 仓库。
- 审计导出副本：`rabbit02:/work/zhanghc/Myllm/SELD/code_releases/SELD_reaudit_20260904_r1`。
- 原始诊断：`rabbit02:/work/zhanghc/Myllm/SELD/experiment_einv2_reaudit_20260904_r1/audit_before.json`。
- 仓库内诊断程序：`scripts/audit/einv2_reaudit.py`。
- 新runtime、train-only scaler和train/eval分离入口已部署并完成smoke；目前尚无新的完整90-epoch实验结论。

### 正式队列已启动

- rabbit02 supervisor PID：`171028`；5张GPU并行，12个90-epoch训练run。
- 启动批次：GPU0–3分别为C0/C1/C2/C3 seed2026，GPU4为C0 seed2027；其余7个排队。
- [启动快照](reaudit_20260904/launch_snapshot.json)、[preflight](reaudit_20260904/preflight.json)、[scaler文件清单](reaudit_20260904/scalar_manifest.json)、[smoke独立validation](reaudit_20260904/smoke_validation.json)。启动快照不是实时状态。
- 实时状态：`rabbit02:/work/zhanghc/Myllm/SELD/experiment_einv2_reaudit_20260904_r1/queue_status.json`；逐run日志在同目录`queue_logs/`，checkpoint/config在`runs/`。

## 代码证据

| 项目 | 结论 | 证据强度 |
|---|---|---|
| 训练与 inference 归一化 | C0–C3 的训练端重排 IV scaler，独立 inference 未重排；C0 实际 scaler 的 mean/std 最大差分别为 0.01353297/0.01276622 | VERIFIED：源码和实际 scaler 构造两端对象验证 |
| scaler split | 旧生成代码的 `BaseDataset` 枚举全部 600 个 dev 文件，没有 train_fold 过滤；scalar.log 记录该生成入口；HDF5 未记录 folds 属性 | PARTIALLY VERIFIED：生成路径支持含 fold1，缺生成当时不可变的完整文件清单 |
| 公共模型初始化 | 固定 seed2026，C0–C3 公共 state_dict 最大差均为 0 | VERIFIED：直接张量比较 |
| 训练随机路径 | 模型构造后的 RNG 状态四组不同；强制同一 forward RNG 后 C2/C3 与 C0 的 SED/DOA 最大差为 2.3227279，C1 与 C0 为 0；C2/C3 先执行 DOA Transformer，C0/C1 先执行 SED | VERIFIED：源码和合成输入测试；不代表此差异解释了历史效果大小 |
| 原始代码版本 | 检查的 EINV2 Python 源文件与此前 source_snapshot 中 SHA256 全部相同 | VERIFIED：实时服务器哈希，见 audit_before.json |
| teacher 轨道对齐 | C2/C3 复用 online tPIT 对 teacher latent 排列，teacher 未独立匹配；不同权重模型不保证同一 slot assignment | VERIFIED：实现事实；训练影响尚未实测 |
| checkpoint 与 metric | 历史结果是旧 checkpoint 的官方口径重评分，不等于已按新官方口径训练及选优 | 已有历史报告；本轮尚未完成全流程回归 |

关键历史路径（均相对服务器实验根目录）：

- `EINV2/C0p1_CausalEINV2_fix_seed2026/seld/methods/ein_seld/{training,inference}.py`
- `EINV2/C0_CausalEINV2_seed2026/seld/methods/data.py`
- `EINV2/C0_CausalEINV2_seed2026/logs/scalar.log`
- `EINV2/C0_CausalEINV2_seed2026/_hdf5/dcase2020task3/scalar/foa_causal_logmel&intensity_sr24000_nfft1024_hop600_mel256.h5`
- `EINV2/C{2,3}_*/seld/methods/ein_seld/models/seld_causal.py`（准确源文件名见原始 JSON）

## 时间对齐：实测结果，不预先替用户决定口径

原始 waveform 采样率 24 kHz，STFT hop600=25 ms，左填充1024。feature k 最晚读取原始 sample `600*k-1`。两次时间池化使标签帧 j 最晚使用 feature `4*j+3`，即读取到 `100*j+75 ms` 之前。

对第4帧（标签帧起点400 ms）的输出：

| 扰动音频起点 | 第0–3帧最大变化 | 第4帧最大变化 |
|---|---:|---:|
| 400 ms | 0 | 0.01630035 |
| 475 ms | 0 | 0 |
| 500 ms | 0 | 0 |

四个 causal variants 的该测试结果相同。测试使用合成0.8 s音频、eval模式、未训练模型；证明了测试中的依赖边界，不等于真实流式部署验证。

- 相对标签帧起点，它有75 ms look-ahead，不能宣称 waveform-level zero look-ahead。
- 若明确允许在标签时间后75 ms或更晚输出，它可以被实现为固定延迟的在线系统；不能通过改名或改标签时间隐藏该延迟。
- 没有用到下一100 ms标签帧的音频，不等同于相对当前帧起点零前视。
- 现有推理按4 s独立chunks处理，没有流式缓存和端到端实时测量，75 ms只是模型的最晚输入依赖偏移，不是实测系统总延迟。

## 同类工作核对

| 工作 | 已核实做法 | 对本项目的意义 |
|---|---|---|
| [Conv-TasNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC6726126/)（语音分离，非SELD） | causal卷积配合只统计当前及过去的cLN；仍按短音频帧处理，论文实验含2 ms帧长 | causal网络不代表零分帧等待或零计算延迟 |
| [Nagatomo et al., On-line SELD, 2022](https://waseda.elsevierpure.com/en/publications/on-line-sound-event-localization-and-detection-for-real-time-reco/) | 明确研究系统延迟与精度关系；摘要报告低延迟下SED下降、定位可保持或改善 | 与本项目关注LE/LR和SED代价有关，但不可用来证明我们的下降原因；本轮仅核实摘要，未核实具体padding实现 |
| [Yeow et al., Real-Time SELD, 2024](https://arxiv.org/html/2409.11700v1) | 从buffer取最近n个音频块；区分采集周期Tr、上下文Tw和处理时间；部署实验Tr=1 s、Tw=2 s | online/real-time是系统执行口径，不能自动推导出每个标签帧零前视 |
| [SELD-TCN, EUSIPCO2020](https://www.eurasip.org/Proceedings/Eusipco/Eusipco2020/pdfs/0000016.pdf) | 原文明确把TCN卷积改成non-causal以利用未来上下文 | 不应仅凭TCN名称把它作为strict-causal baseline |

[DCASE2020官方页面](https://dcase.community/challenge2020/task-sound-event-localization-and-detection)规定100 ms标签/评测分辨率，但并未在这些规则中要求系统必须在帧起点立即输出。

## 建议与待决事项

研究主问题仍是：同等输入上下文及延迟预算下，velocity/latent辅助监督能否改善LE/LR、同时控制SED损失。不要把“降低到零前视”未经讨论地增加为新的主要目标。

1. 无论口径如何，先修复训练/inference归一化不一致，生成仅使用folds2–6的scaler，并统一随机路径及官方validation checkpoint选择。
2. 用户已选择当前帧口径，保留并报告75 ms输入依赖；本轮不采用严格帧起点0 ms，不额外引入pooling改动。
3. 时间池化改变必须从头训练，不能只在旧checkpoint上替换pooling并将结果当公平比较。
4. 保留历史结果；新 C0–C3 × seeds2026/2027/2028 使用独立目录，由有限队列调度，进度以服务器状态文件为准。

目前没有新训练结果，不能声称修复已改善LE或解释了历史差异。
