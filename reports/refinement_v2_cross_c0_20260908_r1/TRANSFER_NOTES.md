# 跨 C0 来源传输记录

## Material Passport

日期：2026-09-08。Origin Skill: experiment-agent。阶段：授权后来源准备，不是训练结果。

仅读取RB05对应seed2027/2028的best.pth、config.json、已有validation参考输出，以及既有scaler的hash。两个权重在源端hash与预登记一致；scaler hash为e3967bac4bd4bc72f57b491c9f1f3a7ede36a65bb244b061b3d6914e75d38d44，与rabbit02已存版本一致，复用现有副本。

公网经本机中转传输过慢；内部服务器间认证不可用。进一步检查发现本机保存的rabbit02-vpn条目禁用了公钥认证，使用本次连接参数显式启用后成功接入同一rabbit02。没有修改全局SSH配置、没有复制私钥到服务器或转发认证代理。

已停止本轮3个慢速传输及其专属连接进程，保留未完成字节文件为best.localrelay.partial.pth和best.stream.pth，不将它们作为C0使用。此为文件传输路径切换，不是实验失败重跑。权重通过本机内网中转重新只读复制；接收端以预登记SHA256完整核验后才能使用。

RB05未执行特征导出、模型推理或训练。所有后续计算仅限rabbit02，空闲3090每次重新核对，不中断其他任务。
