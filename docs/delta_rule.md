# Delta 规则（Gated DeltaNet）

KDA 的线性层用的是 **DeltaNet**：不是简单累加，而是「先删后写」的 Delta 规则

    S_t = S_{t-1} - β · (S_{t-1} k_t - v_t) k_t^T

- β ∈ (0,1) 是写入强度（sigmoid 输出）
- 语义：先按 key k_t 把旧值从状态里**擦掉**，再写入新值 v_t
- 好处：同一个 key 被重复写入时，旧值被覆盖而不是无限叠加

## 为什么能块并行

Delta 更新是**仿射**变换 `S_t = S_{t-1} A_t + b_t`，其中

    A_t = I - β k_t k_t^T        (d_k × d_k)
    b_t = β v_t k_t^T            (d_v × d_k)

两个仿射变换复合仍是仿射：`(A1,b1) ⊕ (A2,b2) = (A1·A2, b1·A2 + b2)`。这正是 `core/scan.py` 用的幺半群，所以 DeltaNet 也能并行前缀扫描。

## 门控

Gated DeltaNet 在写入前给 value 乘一个门 `v_t = v_t ⊙ σ(g_t)`，让模型决定「这一步到底要不要写进长期状态」，抑制无关 token 污染状态。
