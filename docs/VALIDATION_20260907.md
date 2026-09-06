# 0.3.0 验证记录

2026-09-07，整理目录与RB05新建release目录；没有训练真实数据、没有修改旧run。

| 检查 | 环境 | 结果 |
|---|---|---|
| `tests/test_release_030.py` | 本地Python3.12 / 无GPU | 4 PASS：三runtime恢复、当前source一致性、8recipe dry-run、结果重算 |
| `tests/test_repository.py` | 本地Python3.12 / 无GPU | 5 PASS：全目录Python语法、旧入口、16配置dry-run、训练/评价分离、旧来源hash |
| `tests/test_experiment_catalog.py` | 本地Python3.12 / 无GPU | 5 PASS：历史报告重算、链接、6个SVG与来源记录 |
| `tests/test_data_inventory.py` | 本地Python3.12 / 无GPU | 5 PASS：历史数据清单内部一致性；不代表新下载核验 |
| `tests/test_einv2_audited.py` | RB05 EINV2现有环境，`CUDA_VISIBLE_DEVICES=` | 7 PASS：当前帧因果、梯度路由、归一化、PIT/teacher、初始化与EMA |
| `tests/test_einv2_weightprobe.py` | 同上 / CPU | 2 PASS：只改变JEPA系数；输出不变、梯度缩放 |
| `tests/test_einv2_dynamicmask.py` | 同上 / CPU | 2 PASS：norm0等价、moving pair mask |
| `evaluation/test_alignment.py` | 同上 / CPU | 21 PASS：固定官方源码、LR/LF、macro/micro、时长与多声源匹配 |

共 **51项测试通过**。标准Transformer nested-tensor性能警告不影响这些检查。

报告QA：68个主run、136个预期run×split、131个实有评分、5个缺测；逐行SELD公式、唯一键、90epoch日志、checkpoint存在/大小、84行旧证据路径检查通过。`manifest.json#params`按JSON字段定位解析，不误报文件缺失。

三份runtime按训练config保存的source hash逐文件核对；文件恢复字节一致，不代表跨机器训练逐位可复现。检查checkpoint仅存在/大小，本轮未批量反序列化，也未重新推理/评分全部音频。旧项目归档做语法和来源保存，未逐项目端到端训练。

不是论文有效性或互补机制证明：样本量只有三个seed、一个固定validation fold；需要分层分析与评价缺口见结果报告。
