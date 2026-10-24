# 配方：迁移到生产

1. 把 `config.declared` 换成真实规模（2.8T / 896 专家 / 1M 上下文）。
2. 后端从 `cpu` 切到 `vllm` / `transformers`（已预留 adapter）。
3. 用 `deploy/k8s` 或 `deploy/helm` 部署，`/health` 做探活。
4. 打开 `serving.scheduler` 的专家复用批调度。
