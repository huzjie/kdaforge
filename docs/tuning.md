# 调优指南

## 三个杠杆

1. **KDA/MLA 比例**（`attention.kda_layers` / `mla_layers`）：KDA 层越多越省内存但长程精确交互越少；官方是 69/24。
2. **负载均衡强度**（`moe.aux_loss_weight`、`moe.bias_correction`）：吞吐敏感就调高，质量敏感就调低。
3. **投机草稿长度**（`serving.draft_len`）：K 越大单轮吞吐越高，但接受率低时浪费越多。

## 配置档位

- `configs/long_context.yaml`：更多 KDA 层 + 更大 chunk，压内存
- `configs/balanced.yaml`：更强 aux loss + 偏置校正，压热专家
- `configs/highspeed.yaml`：更长草稿 + 更大预取，提解码速度

## 经验

- 先跑 `python -m kdaforge bench` 拿到真实数字再改，别手写估算。
- `python -m kdaforge doctor` 每次改完都要跑，确认 forward 和内存口径没坏。
