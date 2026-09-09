# 唯一v2：结果冻结与有限跨C0确认登记

## Material Passport

- 日期：2026-09-08；Origin Skill: experiment-agent；阶段：freeze / evaluation / conditional cross-C0 confirmation。
- Verification Status: REGISTERED_NOT_EXECUTED。登记发生在本轮新evaluation结果生成或查看之前。
- 唯一v2不变；旧12个小头训练与既有bootstrap不重复。本登记不是新方法方案，也不是额外训练的自动授权。

## A. 第一优先：固定C0的头seed稳定性之evaluation

固定audited RB05 cohort C0 seed2026（SHA256 `8d9c129ae51d847148d44a636aa71fa9359dee23db8886cce9b60132b04511a8`）。先复制并hash冻结12个validation-best权重、完整配置、实际停止/选优记录与解码代码，再导出同一C0 evaluation特征。3个固定对照F0/F-EMA/F-KF，加head seeds2026/2027/2028各F-Deriv/R0/R1/R2，共15个evaluation条件全部保留。

evaluation来自相同TAU2020 FOA数据集，完整200条60秒录音；以实际文件清单/hash核验，不以预计数量代替完成记录。历史evaluation已经被查看过，不能称新盲测。不得按evaluation挑seed、换checkpoint、改网络/损失/超参数/阈值。只保留validation选出的权重；评价脚本没有优化器。

## B. 有限跨C0配对确认（待本次另行明确授权）

| 固定C0 seed | 同cohort既有权重SHA256 | 小头seed | 需要新增训练 |
|---|---|---:|---|
| 2026 | 8d9c129ae51d847148d44a636aa71fa9359dee23db8886cce9b60132b04511a8 | 2026 | 0，复用已有4个validation-best |
| 2027 | 1f924b811f8cc8ed3cfde60d9727f54bb83e950b47404cfa09bcbc694c8aa08c | 2026 | F-Deriv/R0/R1/R2各1次 |
| 2028 | 5d128ad298b110d3afff8ae95fc37b08d25e38fb9d1f595e8d51d99708c76683 | 2026 | F-Deriv/R0/R1/R2各1次 |

权重身份取自既有cache_audit.json中同cohort记录；使用前必须对实际文件hash再次核验，不用rabbit02其他C0替代。若权重、原runtime、scaler或数据无法获得/验证，报告缺项并停止对应条件。RB05不执行推理训练；任何额外只读复制也纳入本次明确确认范围。

最多新增8个正式小模块训练，不重训C0，不复训原有头seed，不添加smoke训练条件或失败自动重跑。先做无优化器的输入/梯度/零修正回归检查；失败只报告，修复或重跑需单独记录与授权。所有运行只使用rabbit02重新核对空闲的3090，最多4个并行小头进程，每个正式训练硬限4小时；非超时异常不自动杀任务。

各C0分别导出自己的train folds2–6和validation fold1冻结特征/原始输出/固定GT目标，绝不共用另一C0的h或原始匹配。新头从头初始化，head seed固定2026。各自validation选优后先冻结，再做evaluation；不得用A阶段evaluation变化修改B阶段设置。

## C. 沿用实际执行的数值策略

- F-Deriv：512→128→3；R0：515→128→3；R1/R2：1031→128→3；GELU，末层零初始化。参数数66,051/66,435/132,483/132,483。
- AdamW lr=.001、weight_decay=.0001、batch32个40帧chunk、梯度clip1、FP32，无scheduler、augmentation或lambda扫描。
- R0/R1固定匹配方向损失mean(1−dot)；R2加lambda=1的按坐标均值单位方向位移MSE；未匹配正则lambda=.1。F-Deriv有效pair SmoothL1(beta=.1)，未匹配零位移正则lambda=.1。
- min40/max200 epochs；验证SELD_LR连续20轮未改善超过1e-5，且相邻两个10轮训练loss均值相对差≤1%，两者均满足才提前停止。每轮完整validation，严格最小SELD_LR选best，平分取较早epoch。停止点不替代best。达到上限而未双平台时如实标未确认收敛。
- 解码保持原活动/类别/概率/实例条目，只替换方向；threshold>.5，raw预测关联45°门限、1e-6°近并列处理；4秒chunk重置，无GT输入关联、无修正方向反馈或修正后去重；rho=15°；F-Deriv/F-EMA alpha=.5非递归两帧，F-KF原登记常速度参数不变。
- 指标固定DCASE2023 micro为主，同时保存实际评分器输出的所有协议数字。纯SED计数单独核对，不将空间F20或LR_CD当作纯SED。

## D. 报告、诊断与计数分离

主配对比较固定：R0−F0、R0−F-Deriv、R1−R0、R2−R1，并报告F-EMA/F-KF。全部官方ER20/F20/LE_CD/LR_CD/SELD、逐录音计数、检测保持、checkpoint/runtime/scaler/代码/数据/预测hash全量保留，不以效果大小过滤。

诊断保留每个C0自身raw类内Hungarian固定匹配（无20°筛选）的全部分母、未匹配预测/漏检数、有效motion pair和排除原因；固定目标静态pair/运动pair/无有效pair均报告。另复用历史GT连续同身份segment静态/动态定义（任一相邻角位移>1e-6°整段dynamic；至少2帧且不动为static；单帧unobservable），与pair分层分开，不重定义阈值。GT属性只用于离线诊断，不能修补推理关联。

交付分为两个独立表：

1. 固定C0头seed稳定性：1个C0×3个头seed×4条件，已有12次训练；A阶段15个evaluation条件。
2. 跨C0权重配对结果：3个C0×固定head seed2026×4条件；复用4次旧训练+最多8次新训练。各C0自有F0/F-EMA/F-KF对照。不得把前一表的头seed当作额外C0或混合计算sample SD。

不重复已有bootstrap；本阶段先交付全部逐C0配对差与均值/sample SD，无自动显著性检验。任何缺失/失败/未授权项写明状态，不填预计成绩。原数字晋级阈值不自动改写；不声称独立新盲测或SOTA。
