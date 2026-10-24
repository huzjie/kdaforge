# 配方：压平热专家

```yaml
moe:
  aux_loss_weight: 0.05   # 更强负载均衡损失
  bias_correction: true   # 逐专家偏置校正
```
验证：`python -m kdaforge bench` 看 `load_balance` 的 imbalance 从 skill0.5 的
高值降到 skill1.0 的 ~1.0。
