# 版本管理

- `main` 保存经过检查的代码。每次改变模型、loss、数据切分或metric须单独提交；不要在一次提交里混入无关实验。
- 新方法使用 `method/<topic>`，实验配置使用 `experiment/<variant>` 分支；本轮不替用户创建无关分支或改仓库保护设置。
- 版本号在 `VERSION`，变化在 `CHANGELOG.md`。每次实验记录 commit SHA、dirty diff、config、seed、data/scalar标识、checkpoint与metric protocol。
- 本次以两类提交分开导入：实际模型/metric源代码；便携入口/配置/文档/测试。保留原仓库初始历史，不 force push。
- 数据、features、HDF5、checkpoints、完整预测与训练日志留在实验服务器；Git仅保存代码及小型结果索引。
- `historical/` 是只读来源记录，不能编辑成当前结果。更正结果新增版本，并写清metric/checkpoint选择变化。
- 当前首版保留EINV2独立变体代码。未来若去重，需先验证forward/checkpoint/aux行为等价，不能仅按类名合并。
- 本地整理目录不是原运行目录；rabbit02的旧训练项目不做重命名、清理或原地替换。
