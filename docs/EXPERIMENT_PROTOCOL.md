# 当前实验协议

整理版本不改模型层、loss 数学或已训练权重。原代码 hash 与所作路径/入口改动分别见 `provenance/source_snapshot.json`、`provenance/relocation_changes.json`。

| Family | Variant | 方法 | 主配置 |
|---|---|---|---|
| EINV2 | C0 | 修正 IV 顺序、小初始化 DOA head、bestfix 的 causal baseline | `configs/einv2/C0.yaml` |
| EINV2 | C1 / C2 / C3 | velocity / JEPA / 二者联合，aux weight 0.2 | 对应 `{variant}.yaml` |
| EINV2 | E0 / E1 / E2 / E3 | offline baseline / velocity / JEPA consistency / 联合 | 对应 `{variant}.yaml` |
| Multi-ACCDOA | C0 / C1 / C2 / C3 | causal baseline / velocity / JEPA / 联合 | 对应 `{variant}.json` |
| Multi-ACCDOA | A0 / A1 / A2 / A3 | noncausal baseline / velocity / JEPA / 联合 | 对应 `{variant}.json` |

Multi portable configs 的非零辅助权重为0.05；`parameters.py` 的历史默认0.2被入口显式覆盖。A1/A2 配置存在不表示完整三 seed 实验已经完成，结果以 reports 中实际 run 为准。

## 固定设置

- TAU2020，FOA，14 classes；train folds2–6，validation fold1，evaluation 200 clips，每条60s；label hop100ms。
- EINV2：24kHz，256 mel，hop600，4s chunks，90 epochs，batch32，Adam lr0.0005，tPIT，SED/DOA各0.5。
- Multi：24kHz，64 mel，5s labels，100 epochs，batch128，lr0.001，ADPIT。所有 A/C 当前 paper tasks 共用 strict-causal frontend、仅train folds拟合normalization。
- EINV2 JEPA horizon 1/3/5 frames，EMA0.996；当前forward为不同独立变体类，不假设关闭aux即可精确等价baseline。

## 缓存与 causal 限制

EINV2 causal historical scalar 按旧IV通道顺序保存，训练/推理代码按 `foa_iv_acn_reorder` 重排；**不要再给它一个已经重排的 scalar**。offline E0–E3保留 `legacy_foa_iv_order` 和对应legacy scalar。`--scalar-path` 明确选择输入文件，但不会判断它的语义是否正确；复现实验应匹配历史路径/hash。

`--hdf5-root` 下应有 `dcase2020task3/data/24000fs/`、`meta/`、`scalar/`。velocity/jepa roots 是直接包含 `dev/` 的目录，不是dataset总目录。辅助 targets 只供训练/validation losses，不作为在线模型预测输入。

EINV2 的既有代码保留 `cudnn.benchmark=True`；不能据 seed 宣称跨硬件bitwise deterministic。此版不借目录整理静默改变原training算法。CPU synthetic feature-prefix 测试不替代完整audio时间戳/端到端causality验证。

## Evaluation

共同的 `dcase2023_micro_full_duration`：官方2023 matching/counting，micro，20° location-dependent detection threshold，完整60s、1s blocks。
LE_CD 是匹配后的角度误差；LR_CD 是定位 recall，不用20°筛选匹配。不同版本不仅LR不同，LE/ER/F也可能变。

旧DCASE2020代码第四项是LF_CD；为历史复现保留，不能改名成recall。宏平均另报；不把F20当纯SED F1或mAP。旧motion分层结果保持自定义诊断标识。

新训练按统一metric选择checkpoint，历史结果则对旧checkpoint重评分；二者是不同provenance。固定checkpoint做指标重算与重训不能混成一个实验。
