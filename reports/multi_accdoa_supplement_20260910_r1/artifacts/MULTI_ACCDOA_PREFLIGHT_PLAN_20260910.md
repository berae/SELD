# Multi-ACCDOA补充：只读核查与待执行预算登记

## Material Passport

- 日期：2026-09-10，Asia/Shanghai。
- Origin Skill：academic-research-suite / experiment-agent，plan。
- 状态：只读来源核查已完成；前向回归、适配代码与实验尚未执行，不是VERIFIED。
- 当前授权：用户确认先做只读核查和适配预算登记，再确认开跑。本文件不构成新增计算授权。
- 交接文件：MULTI_ACCDOA_SUPPLEMENT_HANDOFF.md，SHA256 `f4fb7596cdd875554948f40d2520ac6e1ad5d4441b0225984b82bc34deae85b8`。
- 本地发布副本HEAD：`f1cad40a64ce7df5735ba024a72b1fad9e73ca67`，核查时工作树干净。本轮不修改/推送该仓库，不更改既有EINV2实验。

## 1. 已确认的来源

远端项目根为 `/work/zhanghc/Myllm/SELD/DynamicCausalMultiACCDOA_TAU2020`，以下路径相对此根。

| 对象 | 路径 / 证据 | SHA256或状态 |
|---|---|---|
| 指定C0 seed2026 | `runs/models/35_paper_v1_C0_seed2026_eval_split0_multiaccdoa_foa_model.h5` | `8c0fc23d13f35468d69b9e35e96a3868011b06384f9de9196ac06f392f40bcc1` |
| 历史manifest | `runs/manifests/35_paper_v1_C0_seed2026_eval_split0_multiaccdoa_foa.json` | `0f68bb78e90f00dbc5311b62462bbfc796efed160ce4b37340c1cdb31d35940f` |
| scaler | `features/tau2020_foa_multiaccdoa_strictcausal_trainfolds2-6/foa_wts` | `35f60fdb0d7467e005a8f530a0786a29057e8d160e88b409c4cd26e9cd2a1b27` |
| 模型源码 | `seldnet_model.py` | `99790739f0e4ab994d413f07a945c1cd648c845fc1b23fb22684a58346037a2e` |
| 生成器 | `cls_data_generator.py` | `01859e2c95b2ddd11292f6cbb4ade850a782ed3aad0ddb7234bc1e73e5e76514` |
| 特征源码 | `cls_feature_class.py` | `901beacc230d5874c9e50e414391f330d1bc6782966ecacd6598a52915fd3cb5` |
| 历史训练入口 | git历史对象中的`train_seldnet.py` | `2b3821935026e1e5d90df9818b888423fedc3e5dc78d2c4712258c6e6b84dc1a` |
| 当前训练入口 | `train_seldnet.py` | `57b4284a1ce39420de455de9686e58b0ffd3db1b3dabb1b5a144cf3d4f12f37c` |

历史提交 `ca05f674b3c719c7b1912417a766f3219197e1a1` 仍可读取；远端当前HEAD为 `560a17a59192971b5c2c241ef4caea82f8ebb8e8`。模型、生成器和特征源码与历史提交无差异。训练入口增加了validation-only退出分支；当前评分文件另有未提交的指标对齐改动。不能直接将整个当前工作目录声称为历史原封运行时，后续应从历史对象提取独立快照，评分协议另行封存，不回退或覆盖用户目录。

历史启动脚本指向 `.venvs/dcase2023-official-py38/bin/python`，当前该解释器报告Python 3.8.11，安装目录标识PyTorch 1.10.0+cu111。这是现有环境证据，不是历史依赖锁定的完整证明；尚未完成此环境的模型加载/前向可用性验证。

仅在CPU上用另一现有PyTorch环境的`weights_only=True`读取state_dict形状，没有构造模型或执行前向。权重中末层形状为126×128、前一层128×128，确认末层输入128维；126=3轨道×3坐标×14类。hook输出和时间轴形状仍待前向验证。

## 2. 缓存与历史回归参照

- 原声学dev归一化特征600份，fold1至fold6各100；evaluation特征200份。scaler旁记录fit_files为500份、fit_splits为2–6。
- 历史validation CSV有两套，各100份，内容不同；历史evaluation CSV为200份。
- 在本项目runs树内未找到npy/npz浮点输出缓存；不能据此声称其他目录绝无缓存。本次未发现可直接复用的小头隐藏特征缓存。
- 所有以上为文件/源码核查，未重新导出或评分，也未执行任何bootstrap。

### validation差异的原因与正确参照

原目录 `runs/results/35_paper_v1_C0_seed2026_eval_split0_multiaccdoa_foa_20260902091211_val` 在训练循环中每轮被写入，而权重只在validation改善时保存。日志最终轮99、最佳轮82。因此，原目录不是可直接要求最佳权重逐字节复现的validation参照；已观察到的差异不能被直接定性为新导出失败。

应核对的最佳权重validation历史参照是：

`runs/analysis/threshold_predictions/Completion_20260904_paper_v1_C0_seed2026_validation_seed2026_validation_t050`

其2026-09-04完成回执位于 `/work/zhanghc/Myllm/SELD/experiment_completion_20260904/runtime/Completion_20260904_paper_v1_C0_seed2026_validation.json`，记录用历史Python环境执行`evaluate_threshold_sensitivity.py`、阈值0.50、退出码0。该参照已由本地历史索引使用；后续需进一步封存该次manifest、实际解码脚本与文件hash，验证其确实绑定本次权重，不能仅凭目录名确认。

原始evaluation参照为 `runs/results/35_paper_v1_C0_seed2026_eval_split0_multiaccdoa_foa_20260902095315_test`。历史evaluation已暴露，后续不是盲测；新evaluation只在三个头全部冻结后执行。

待通过：未加hook原生前向与历史CSV的比较；加hook前后原输出逐位一致；零修正后Cartesian CSV和统一评分一致。不存在历史浮点缓存时，只能将本次未加hook前向封存为本次浮点基准，不伪称历史浮点回归通过。若发现不一致，保留证据并停下，不放宽门槛或自动重跑。

## 3. 最小适配范围（未实施）

1. 独立导出器：保留原5秒/50标签帧chunk、每录音批处理及padding；通过末层线性层pre-hook读取128维输入。不运行原训练入口，不启用速度/JEPA。冻结计算的实现必须兼顾原生数值一致性，参数和缓冲区前后hash核对，优化器只含小头。
2. 原解码器适配：活动阈值0.5、合并阈值15°及三轨道合并分支/加法顺序原样保留。输出显式事件列表，不使用EINV2的argmax检测、双slot或概率张量假设。
3. 事件键：`recording/frame/class/within_class_ordinal`；另存全帧写出序号、chunk、合并成员、原xyz及范数。CSV中track=0不是唯一ID。已解码事件mask固定，即使合并向量范数低于0.5也不得删除。
4. Cartesian写出保留七列 `frame,class,0,x,y,z,0` 与原float字符串规则。方向修正后不再阈值判断、合并、排序去重或反馈到关联。
5. 历史关联使用原始事件、同类相邻帧、45°门限、原近并列处理；不跨chunk，GT不参与推理。匹配/损失适配到可变事件数，不能复用EINV2活动判定。
6. 三头分别独立初始化，head seed2026；方向修正131→128→3，两个历史版本263→128→3；GELU、末层零初始化、15°修正上限。不得偷偷增加embedding、升维网络或独立运动预测对照。
7. 复用既定单位方向损失、未匹配正则0.1、有效pair位移坐标MSE权重1；位移单位每100ms。AdamW lr0.001、weight_decay0.0001、batch32原生chunk、FP32、clip1，无scheduler/增强/搜索。
8. 40–200轮；validation连续20轮未改善超过1e-5且相邻10轮loss窗口相对变化≤1%才提前停止；按完整validation的dcase2023_micro/SELD_LR严格最小值取最佳，平分取早。最多200仍不满足则如实记收敛未确认。

回归必须覆盖零头恒等、检测条目/重复数、纯SED计数、原始输出封存、未来扰动前缀、chunk重置、同类多源、轨道切换、无前驱/无有效pair、padding及零范数。有效运动预检应检查固定训练录音整段，不要求全静态切片产生非零运动梯度。原前端right-edge时戳与池化可用时间必须核算，不能以causal字段替代测试。

## 4. 资源快照与预算提案

2026-09-10约10:00 rabbit02五张3090：0–3号均被PID3233184占用约20GiB，4号0MiB/0%。这是瞬时快照，不是资源预留。只使用真正空闲卡，不共享该进程、不停止他人任务、不在RB05计算。若仍仅一张空闲，三个头串行；若新增空闲卡，最多三头各一张，全项目相关小头合计不超过四个。每次启动前重新核对GPU UUID、进程与共享任务占用，不能依赖旧launcher充当全局锁。

以下均为待确认的硬上限，不是耗时预测；本轮未启动任何阶段。

| 阶段 | 固定范围 | 单任务硬限 | 最大任务数 |
|---|---|---:|---:|
| 最小运行时/解码预检 | 固定train和validation各一条完整录音；无优化器，含hook对照及合成边界 | 30分钟 | 1 |
| train/validation缓存 | train按fold2–6各100条；validation fold1的100条；每份录音完整原生chunk | 30分钟 | 6 |
| 全缓存预检和训练前封存 | 三个零头、梯度、历史CSV/评分及检测/因果回归；无优化器 | 30分钟 | 1 |
| 三个小头训练 | 每头独立seed2026，40–200轮 | 4小时 | 3 |
| evaluation缓存 | 三头全部冻结后，按固定文件名排序前100/后100条 | 30分钟 | 2 |
| evaluation统一评分/配对交付 | 六条件×200条，逐文件结果与诊断，不新增bootstrap | 30分钟 | 1 |

训练上限12 GPU·小时；其余有界任务累计最多5.5任务小时（含CPU检查/评分，不应全计作GPU时间）。适配编程工时另计，吞吐与显存尚未实测。首次预检若暴露完整缓存会超限，应在全缓存开始前报告，不先失败再自动续跑。任一任务失败/超时不自动重试，已完成结果和原失败均保留。

## 5. 独立路径和交付门槛

拟议远端新根：`/work/zhanghc/Myllm/SELD/reports/multi_accdoa_supplement_20260910_r1`。检查时不存在，本轮未创建；执行时再次检查，不覆盖同名内容。

拟议本地发布结果根：`E:/SELDR/SELD_publish_v2_20260907/reports/multi_accdoa_supplement_20260910_r1`。本轮尚未创建。

子目录拟分为source、preflight、cache/train、cache/validation、heads、frozen、cache/evaluation、evaluation。原模型/源码/配置/scaler/解码器/文件列表全量manifest与hash在首次计算前封存，独立写入。当前没有已实现并通过验证的启动命令，不能冒充可直接运行。

验收不以成绩变好为条件：六条件全部保留，官方指标、逐baseline与逐步差、固定匹配分母、静态/动态诊断、检测保持、逐轮停止/选优及来源齐全；单权重单头seed不报跨seed稳定性。不主张Multi上优于独立运动预测头。与EINV2分表，不把两者绝对分数差归因于结构。

## 6. 当前决策点

只读核查支持进入适配，但不表示可以直接开始三头训练。下一步建议仅授权：在上述独立路径实现最小适配，并执行第一行30分钟、无优化器预检。通过后交回原输出回归、时戳/因果检查、实测耗时与资源预算，再确认全缓存及训练。依academic-research-suite保留“计划/实测/执行授权”的区分，不因赶时间省略回归。
