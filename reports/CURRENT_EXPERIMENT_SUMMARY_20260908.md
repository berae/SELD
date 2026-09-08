# SELD 当前全部实验总览（2026-09-08）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；Origin Mode: validate。
- Origin Date: 2026-09-08；Version Label: current_experiments_r1；Verification Status: ANALYZED。
- 来源：仓库与本地既有逐 run、逐录音、训练停止、冻结和失败记录；本次只整理发布，没有训练、推理或重复 bootstrap。
- 范围是已归档/检索到的实验，不声称穷尽未留证据的历史试跑；无人类逐项已读声明。

## 1. 当前结论与计数

唯一 v2 继续保持：冻结同 cohort 的 C0，只训练保持检测输出的方向修正模块；F-Deriv 使用同 C0 特征重新拟合，R0/R1/R2 是递进消融。旧融合 oracle 不是前置条件，不扩 JEPA、lambda、主干或第二数据集。

| 阶段 | 实际完成及范围 | 当前结论 / 限制 |
|---|---|---|
| 历史统一主矩阵 | 14组×3 seeds=42主干 run，84个 run×split | EINV2 causal/offline、Multi-ACCDOA；旧权重选优后统一重评分，不能当新版重训 |
| 重审后的两个 cohort | 26主干 run；早期总表21个evaluation，后续补齐另5个 | 与历史42组成68个主干 run；两机器cohort不混成6 seeds |
| 固定权重运动拆分 | 18个 run×split 浮点导出、48个解码评估；另5个既有权重evaluation | 都是复用权重，不新增主干训练；5个补评属于前述26个run |
| 唯一 v2 首轮 | C0seed2026、headseed2026，F-Deriv/R0/R1/R2共4个小头 | 已收敛并按validation选优；是下行12个的一部分，不重复计数 |
| 固定C0头seed稳定性 | 增加headseed2027/2028的8个小头，合计12个 | 三头seed检验初始化/shuffle敏感性，不是三套C0 |
| 固定C0冻结evaluation | 3固定对照+12学习条件=15条件，每条件200录音 | 全部完成，保留3,000份最终预测CSV；不是15个新训练 |
| 跨C0权重确认 | C0seed2027/2028各train缓存500/500；validation均首条失败 | 新小头0/8；可复用的C0seed2026/head2026组1/3，不计算跨C0均值 |
| 历史旁支/失败 | 7个新版smoke/初始化记录、175条旧manifest/曲线/评测记录等 | 有迁移副本和重复评分，不与68或12直接相加；STARSS仅训练曲线证据 |

计数更新：9月7日旧总表的68主干run、131个已评分run×split，后续5个缺失evaluation已补齐，覆盖应为136个run×split；这只是已有权重的覆盖更新，不是新增5次训练。保留旧表原貌，后补证据见[运动拆分报告](motion_decomposition_20260907/README.md)。不将68个主干run与12个小模块包装为同质的“80次独立验证”。

## 2. 唯一 v2：完整对照摘要

TAU2020 FOA；500 train、100 validation、200 evaluation。DCASE2023 micro是评分协议，不是数据集年份。下表LE（度）/SELD均越低越好；学习条件为三个头seed均值，sample SD及所有ER20/F20/LR_CD详见原报告。固定对照只评一次，不复制成三个独立样本。

| 方法 | Validation LE | Validation SELD | Evaluation LE | Evaluation SELD |
|---|---:|---:|---:|---:|
| F0 | 15.000487 | 0.370666 | 14.342918 | 0.359813 |
| F-EMA | 14.995210 | 0.370765 | 14.327649 | 0.359764 |
| F-KF | 14.988051 | 0.370686 | 14.337258 | 0.359757 |
| F-Deriv | 14.977945 | 0.370671 | 14.307297 | 0.359675 |
| R0 | 14.731054 | 0.369099 | 14.109981 | 0.358422 |
| R1 | 14.770742 | 0.369727 | 14.146786 | 0.358840 |
| R2 | 14.756653 | 0.369426 | 14.093151 | 0.357913 |

R0相对F0和同C0 F-Deriv，validation/evaluation的LE与SELD均3/3头改善。Evaluation相对F0平均LE下降0.232938°、SELD下降0.00139051，量级有限。R1没有稳定额外增量；R2的evaluation平均值更好，但R2−R1的LE仅1/3头改善、SELD为2/3，不能按evaluation事后换候选或宣称运动机制已证明。

### 实际停止与选优

统一min40/max200 epochs；validation连续20轮无超过1e-5改善，且相邻两个10轮训练loss均值相对变化≤1%；严格最小validation SELD_LR选优，平分取早。双平台是操作性收敛，不是全局最优证明。不是固定20轮，也不是末轮权重。

| 头seed | F-Deriv 停止/最优 | R0 停止/最优 | R1 停止/最优 | R2 停止/最优 |
|---|---:|---:|---:|---:|
| 2026 | 151/129 | 81/16 | 79/44 | 83/16 |
| 2027 | 158/74 | 80/3 | 97/15 | 97/52 |
| 2028 | 161/120 | 86/14 | 83/3 | 84/14 |

完整来源：[首轮](refinement_v2_execution_20260907_r1/RESULTS.md)、[稳定性与逐run](refinement_v2_review_20260908/STABILITY_RESULTS.md)、[登记](refinement_v2_review_20260908/PREREGISTRATION.md)、[冻结和实际停止索引](refinement_v2_confirmation_20260908_r1/EXISTING_RESULTS_AND_FREEZE.md)、[15条件evaluation](refinement_v2_confirmation_20260908_r1/EVALUATION_RESULTS.md)。

## 3. 检测、因果与机制诊断

- 15/15 evaluation条件保持原始有序(frame,class,slot)条目，仅替换方向。纯SED TP/FP/FN一致，micro F1=0.6723135176781201、macro F1=0.6555745034419547。空间F20/官方LR变化不等于纯检测输出变化。
- 已执行波形/特征前缀扰动、有限分块与关联回归；不把有限测试称为所有输入的形式化因果证明。
- Evaluation固定原C0匹配：GT117,220、活动预测99,673、固定匹配71,942、未匹配预测27,731、未覆盖GT45,278。固定目标诊断不按修正结果重匹配，且不是官方LE。
- 固定匹配静态/动态source分母38,652/33,290。R2相对R0静态误差均值下降约0.052816°，动态近乎不变且略差0.000526°，不支持动态特异收益主张。
- 原headseed2026诊断显示R2优化了运动pair MSE，但不等于绝对定位改善；不训练的F-EMA运动MSE甚至更低。R0后期训练loss下降而validation回退，说明延长训练不自动解决泛化。
- 既有20,000次录音配对bootstrap只针对原headseed2026的validation与给定选优模型；不包含选模/主干不确定性，不代表独立测试显著性。本次未重跑或新增bootstrap。

来源：[分析与既有bootstrap](refinement_v2_review_20260908/ANALYSIS.md)、[全量固定匹配诊断](refinement_v2_confirmation_20260908_r1/evaluation/fixed_matching_diagnostics.json)。

## 4. 历史探索得到什么

- 历史EINV2/Multi辅助监督存在LE与召回的权衡；新版两个cohort的velocity-only平均LE下降、LR下降。不能仅凭LE证明全体声源改善。
- RB05共同匹配拆分：C1−C0 evaluation平均共同目标误差−0.305±1.115°，seed2028负向，三seed均lost>new；不能将全部收益归因于漏掉难目标，也不能称稳定。
- 既定两帧fusion相对smoothing平均LE差约+0.000135°，没有稳定额外收益。C3_j005在同解码条件下没有显示优于C1的平均LE；JEPA predictor也未稳定超过persistence。
- 后补5个evaluation：C3_j020三seed、D1与D3各seed2026已完成。它们是探索性附录，旧总表的缺失描述已过时，不用于继续选择lambda。
- 旧Multi λ=.2的A1/A2确有三seed，缺的是λ=.05统一主矩阵；STARSS22/23有80轮曲线但无可确认独立test；ObjectStateSELD仅工程smoke。旁支不纳入唯一v2结论。

完整历史表：[68主run及旁支索引](ALL_EXPERIMENTS.md)、[旧42run统一指标](EXPERIMENT_CATALOG.md)、[后续运动作用拆分及5项补评](motion_decomposition_20260907/README.md)。不同cohort、因果/非因果、旧/新选优版本不可直接混合排名。

## 5. 跨C0确认：当前未完成

本次用户已另行授权最多8个小头及RB05只读传输。两份准确权重/配置/历史validation参考已到rabbit02，来源hash核验通过。各自train缓存500条完成，但两份validation均在首条`fold1_room1_mix001_ov1`的SED原始浮点逐位比较失败，写入0/100；新小头0/8，新evaluation未开始。

未放宽比较、未自动重试、未重训C0、未更改结构或lambda。当前没有误差幅度或指标差异可填，不能说“只是舍入误差”。后续首批诊断仍待另行确认；本次上传不恢复计算。

最新权威状态：[STATUS.md](refinement_v2_cross_c0_20260908_r1/STATUS.md)。该目录根部报告中的部分相对链接使用远端原布局；本地完整配置/日志在[artifacts](refinement_v2_cross_c0_20260908_r1/artifacts)。

## 6. 发布范围与时序

本次新增发布：三头seed稳定性与既有bootstrap/目标诊断、全部逐epoch/逐run小型记录、15条件evaluation指标/逐录音计数/3,000份预测CSV、冻结配置与hash、跨C0授权/实际配置/失败日志。权重、浮点特征、音频、压缩包、SSH known_hosts不发布；原始副本不删除。

原报告保持历史时点的“待授权”“未完成”“推送失败”文字，不回写冻结材料。后续顺序是：固定C0登记与评估→用户明确确认跨C0→只读传输与缓存失败→本次结果发布。跨C0当前状态以STATUS为准；仓库最新入口以本总览为准。

大文件在rabbit02 `/work/zhanghc/Myllm/SELD/reports/` 对应目录；本地交付包仍保留在 `E:/SELDR/SELD_motion_20260907/reports/`。归档包链接在GitHub不可下载，不代表原件丢失。新增证据目录禁用Git换行转换，保留来源字节/hash。

## 7. 解释限制与检查（11/11）

| 检查 | 处理及剩余限制 |
|---|---|
| Simpson反转 | 保留静/动态及seed分歧，未穷尽每房间/类别 |
| 生态谬误 | 头seed均值不推断每录音或跨C0泛化 |
| Berkson选择 | LE与固定匹配均有条件选择，保留未匹配分母 |
| Collider偏差 | 不将GT诊断子组用作推理/因果控制 |
| 基率 | 明列全GT、活动、匹配、静动态分母 |
| 均值回归 | 披露validation选优，非末轮、非挑seed |
| 幸存者偏差 | 全部15条件与两次validation失败均保留 |
| 多重比较 | 不新增p值或筛选后显著性主张 |
| 分析路径 | 历史探索、冻结确认与后续授权分开；不按evaluation调参 |
| 相关与因果 | 消融差、运动MSE及局部梯度不是机制证明 |
| 反向因果 | 时序测试只支持其覆盖的推理约束 |

总体置信边界：CAUTION。当前支持固定一个C0下小幅、较一致的方向修正收益；不支持跨C0稳定、动态机制普适有效、SOTA或录用保证。历史evaluation已暴露，不能称新盲测。本轮是归档与描述性汇总，不是完整训练独立复现。
