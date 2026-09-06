# DCASE 2023 官方 Multi-ACCDOA 复现说明

## 复现对象

- 官方仓库：`sharathadavanne/seld-dcase2023`
- 固定 commit：`24d14c9ebe6878062dc3b68eb07dd0100722f530`
- 任务：DCASE 2023 Track A 音频 SELD baseline
- 输入格式：STARSS23 FOA
- 输出：Multi-ACCDOA，使用 ADPIT 处理同类多声源轨道排列

## 公平对照口径

- 训练：DCASE 2022 官方合成数据 fold1/2 + STARSS23 实录 fold3。
- 验证/测试：STARSS23 开发集 fold4。官方仓库在开发集上将 fold4 同时作为 validation 和 test；因此这里的结果不是 hidden evaluation GT。
- 官方 README 公布的 FOA 开发集结果：ER20=0.57、F20=29.9%、LECD=21.6°、LRCD=47.7%。仓库同时说明不同运行环境下不能逐点完全复现，应以接近为目标。
- PSELDNets 作者公开的约 515 GB 合成集单独下载与保存，不混入本轮官方 baseline，除非后续建立明确的“额外数据”实验组。

## 仅做的等价配置改动

- 将官方源码中的作者绝对路径改为 rabbit02 项目路径。
- 新增任务号 `30`：与官方 FOA Multi-ACCDOA 参数一致的 quick smoke。
- 新增任务号 `31`：与官方任务号 `3` 等价的全量 FOA Multi-ACCDOA 复现。
- 保留原始 `parameters.py` 为 `parameters.official.py`，用于逐项审计。

## 执行门禁

1. DCASE 2022 官方合成分卷完成官方 MD5 校验、合并、ZIP CRC 和解压。
2. 数据审计确认 fold1/2/3/4 的音频与标签数量匹配。
3. 官方 Python 3.8.11 / Torch 1.10.0 环境导入通过。
4. 运行官方特征与 Multi-ACCDOA 标签预处理，核对输出不为空且 fold 齐全。
5. 任务号 30 完成 2 epoch / 2 batch quick smoke，loss 与四项指标均为有限值。
6. 以上全部通过后，才启动任务号 31 的 100 epoch 全量训练。

## 当前状态

- 官方源码：已准备。
- STARSS23 实录 fold3/4：已通过符号链接复用，168 条 FOA 与 168 份 metadata。
- DCASE 2022 官方合成 fold1/2：下载中。
- PSELDNets 作者完整合成数据：按当前优先级冻结，已下载的断点文件独立保留，不参与本轮复现。
- 正式训练：未启动。
