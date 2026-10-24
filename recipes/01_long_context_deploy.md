# 配方：1M 上下文部署

```yaml
# configs/long_context.yaml 要点
attention:
  kda_layers: 6        # 更多线性层压内存
  mla_layers: 2
  chunk_size: 32       # 更大块提吞吐
  gate: true
```
效果：KV 内存从 full 降到 hybrid，比例随上下文长度线性拉大（见 `doctor`）。
