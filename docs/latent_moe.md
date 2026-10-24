# Stable LatentMoE：896 专家怎么不打架

2.8T 参数里只有 104B 被激活，靠的就是稀疏 MoE：896 个专家里每次只挑 top-16。但稀疏 MoE 有个著名的坑——**热专家**：门控一旦学出偏好，少数专家吸收绝大多数 token，成为 serving 吞吐瓶颈。

## 两个机制

1. **辅助负载均衡损失**（Switch Transformer 式）

       L = n_experts · Σ_i f_i · P_i

   f_i = 路由到专家 i 的 token 占比，P_i = 门控对专家 i 的平均概率。它惩罚「集中路由」。

2. **逐专家偏置校正**（Stable 的关键）：维护每个专家的可加偏置 b_e，专家过载时就往下调

       b_e ← b_e - lr · (count_e - mean_count) / mean_count

## 复现要点

- `imbalance = max_load / mean_load` 是最直观的度量，1.0 = 完美均衡。
- 偏置校正只改路由、不碰专家权重，是**零成本**的均衡手段。
- 参考实现里 skill=1.0 时 imbalance 压到 ~1.0，skill=0.5 时保持天然热专家偏置。
