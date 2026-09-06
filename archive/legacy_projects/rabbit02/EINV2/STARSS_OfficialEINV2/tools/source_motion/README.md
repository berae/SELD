# STARSS22/23 数据准备与方向运动分层

本目录保存 STARSS 下一阶段基线的数据审计脚本。

## 运动定义

- 分析对象是每个录音内连续出现的 `(class_id, source_id)` 方向轨迹片段。
- 方位角、俯仰角先转换为单位笛卡尔方向向量 `(x,y,z)`。
- 当前速度监督是单位方向向量的一阶差分，因此这里的“动态”专指方向发生变化；只有距离变化但方向不变的径向运动不属于当前速度头的可观测范围。
- 主比较仅使用高置信度 `static` 与 `dynamic`；阈值附近的 `borderline` 和帧数不足的 `insufficient` 单独报告。

默认高置信度规则：

- `static`：轨迹相对平均方向的最大偏差不超过 1°，且 95% 相邻帧角速度不超过 10°/s；
- `dynamic`：最大偏差至少 2.5°，并且至少 3 次相邻帧位移达到 1°，或出现至少 50°/s 的清晰运动；
- 标签分辨率为 100 ms。

## 产物

- `source_trajectories.csv`：source 轨迹级统计和运动分组；
- `source_frames.csv`：逐 source-frame 的方向、笛卡尔坐标、方向速度和运动分组；
- `motion_summary_by_split_class.csv`：按官方 split、类别和运动分组汇总；
- `motion_summary.json`：数据集总体统计、重叠度和判定阈值。

`prepare_canonical_metadata.py` 会为官方 DCASE2022 EINV2 代码生成统一的五列极坐标标签。STARSS23 原始第六列距离仍保留在原始数据中，只在 2023 SED+DOA 基线的数据入口中剥离，避免旧代码把六列格式误判为笛卡尔坐标。

## 静态/动态诊断评测

`evaluate_motion_strata.py` 先在每个 `(文件, 帧, 类别)` 内对全部 GT 和预测执行一次匈牙利角距离匹配，再按照 GT source 的运动分组汇总：

- 20° 内 source-frame recall；
- 未阈值匹配 recall；
- 平均、中位数和 P90 定位误差；
- 微平均与按类别宏平均结果。

无法匹配到任何 GT 的额外预测只计入全局 FP，不强行归因给静态或动态 source。因此该评测用于回答“速度/JEPA 是否更有利于动态声源”，不替代官方 ER20、F20、LECD、LRCD。
