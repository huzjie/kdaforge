# 线性注意力：为什么 1M 上下文不靠 KV cache

标准 softmax 注意力对每个 token 存一份 KV，内存随序列长度线性增长到 O(L·d)，1M 上下文直接爆显存。线性注意力的做法是**把注意力写成循环状态**：

- 状态 S（d_v × d_k）与归一化向量 z（d_k）
- 每步 `S += outer(v, k)`，`z += k`
- 输出 `o = S q / (z·q + eps)`

状态大小与序列长度无关，恒为 O(d_k · d_v)。这就是 KDA 里 69/93 层能跑 1M 上下文的根因。

## 块并行：关联扫描

朴素循环是 O(L·d_k·d_v) 串行。因为状态更新是**可结合（associative）**的，可以用并行前缀扫描（prefix scan）把它拆成块：块内规约 → 跨块扫描 → 块内展开。这正是 FlashLinearAttention 一类内核的思路，本项目的 `core/scan.py` 给了通用实现。

## 复现要点

1. 状态更新必须构成**幺半群**（有结合律 + 单位元）才能块并行。
2. 因果性靠「用前 t-1 个 token 的累计态」实现，等价于「排他前缀」。
3. 参考实现里 `linear_recurrent == linear_chunked`（test_linear.py 断言），说明扫描没有近似误差。
