# 0.1.0 整理验证

执行日期：2026-09-04。验证使用rabbit02既有环境；代码位于独立 `code_releases/SELD_v0.1.0_20260904`，没有替换原训练项目。

| 检查 | 结果 | 范围 |
|---|---|---|
| `tests/test_repository.py` | PASS，5 tests | 全部Python语法、6入口help、16训练dry-runs、导入源hash与改动hash、train/test隔离 |
| `evaluation/test_alignment.py` | PASS，21 tests | 官方源码hash、matching、trueLR/legacyLF、完整时长、jackknife点估计、14类macro等 |
| `tests/test_entrypoint_config.py` | PASS，2 tests / 16 variants | 实际参数与config解析；训练调用被替身替代，没有训练 |
| `tests/test_einv2_variant.py` | PASS，8 variants | C0/C1/C2/C3/E0/E1/E2/E3，CPU synthetic forward；C0–C3 feature-level prefix invariance |
| Multi `test_dynamic_aux.py` | PASS | ADPIT alignment、track-switch mask、aux shapes/causal prefix |
| Multi `test_experiment_protocol.py` | PASS | variant matrix、共同frontend、seed override、train-only scaler、STFT prefix |

首次配置测试中，测试替身清理 `sys.modules` 干扰了重复加载的YAML parser；已修正测试隔离并重跑通过。没有据失败测试发布通过状态。

EINV2测试环境：Python3.10.20，Torch2.4.1+cu121，numpy1.26.4；Multi：Python3.8.11，Torch1.10.0+cu111，numpy1.22.4。依赖文件记录直接依赖版本，非重新安装验证后的完整environment lock。

## 未覆盖

- 未重训，未在新目录完整运行所有checkpoint的推理，也未重新创建全新依赖环境。
- synthetic feature-prefix测试不等于端到端audio全部边界的causality证明。
- `reports/aligned_20260904/` 是此前84预测集重评分证据，不是新结构下重训结果；历史checkpoint仍按旧validation criterion选择。
- RB05未做该整理版本的环境验证或同步部署；没有启动任何训练。

新版本后续正式训练前应先在目标环境运行上述检查，再对所选variant做短小的端到端数据/推理检查。
