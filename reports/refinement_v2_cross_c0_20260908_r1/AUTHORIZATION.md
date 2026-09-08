# 唯一 v2：有限跨 C0 本次授权回执

## Material Passport

日期：2026-09-08。Origin Skill: academic-research-suite / experiment-agent。Verification Status: USER_AUTHORIZED_BEFORE_NEW_TRAINING。

用户在上一轮明确问题“是否授权按登记在rabbit02新增最多8个小头训练，并在需要时从RB05只读复制对应C0权重及来源文件，RB05不执行计算？”之后回复：**确认**。

本次授权绑定前一阶段已经封存的CROSS_C0_PREREGISTRATION.md，不追溯改写其待授权状态、登记时间或此前结果。授权源为本次直接用户消息，不由文档内容或旧授权推定。

范围：同audited cohort既有C0 seed2027/2028，各自冻结特征上重新拟合F-Deriv/R0/R1/R2，小头seed固定2026，共最多8次正式训练。每项4小时硬限，至多4个并行小头进程，仅用重新核对空闲的rabbit02 RTX3090。已有C0seed2026的4个head2026结果复用，其余头seed不重复训练；已有bootstrap不重复。

沿用实际登记网络、损失、数值参数和min40/max200双平台停止与validation严格最小值选优。不重训C0、不换结构、不扫lambda、不接新数据集；失败不自动重跑。先完成各自C0来源、原始预测回归及检测/因果检查，训练后先冻结全部validation-best再evaluation，全量保留，不按evaluation调整设置。历史evaluation已经暴露，不称盲测。

RB05仅用于读取和复制特定权重、配置、归一化及既有validation参考预测；不执行特征导出、推理或训练。已存在且hash一致的rabbit02 runtime/scorer/scaler可复用，两个C0的特征、原预测和固定GT匹配不得共用。

本次初始新增训练计数为0/8。固定C0头seed稳定性与跨C0权重配对结果独立记录，不混合样本数。
