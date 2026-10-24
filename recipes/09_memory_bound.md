# 配方：内存受限场景

在显存吃紧的边缘设备上，优先：
1. 拉高 `attention.kda_layers`（更多循环态层，更少 KV cache）
2. 调小 `attention.head_dim`（d_k*d_v 状态更小）
3. 关掉 `attention.gate`（省门控计算）
代价是长程精确交互变弱，需在 retrieval 基准上验证。
