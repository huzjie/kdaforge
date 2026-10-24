# 贡献指南

1. 保持**零依赖**：核心代码只允许用 Python 标准库。
2. 保持**确定性**：随机一律走 `utils.random` 的 md5 种子，禁止裸 `random.random`。
3. 训练信号**单调趋 1**：新基准的 skill 曲线必须从随机基线单调爬向最优。
4. 改完跑三步：`unittest` → `doctor` → `bench`。
5. 声明规模只写进 `config.declared`，实例化保持 tiny。
