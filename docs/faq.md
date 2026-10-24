# FAQ

**Q: 这是 Kimi K3.1 的官方实现吗？**
不是。是零依赖的**参考实现**，复现其架构思想（KDA / Stable LatentMoE / MOPD / 投机解码），用于学习与原型验证。

**Q: 为什么声明 2.8T 却只实例化 tiny 结构？**
声明规模（`config.declared`）用于口径表达；参考实现实例化 64 维 / 16 专家，让纯 Python 前向毫秒级跑完。要换规模改 config 即可。

**Q: 需要安装 numpy/torch 吗？**
不需要。全部用标准库，Python ≥ 3.8 即可。

**Q: chunked 和 recurrent 结果一致吗？**
一致。关联扫描无近似，`test_linear.py` / `test_scan.py` 断言两者相等（浮点容差内）。

**Q: MOPD 为什么不用外部教师？**
设计目标就是「沙箱自动奖励 + 自生成教师」，不依赖闭源模型 API，也不依赖人类标注。
