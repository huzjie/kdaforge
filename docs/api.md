# API 参考

## 顶层

- `Config` / `load_config(path=None)` — 配置
- `KDAForgeModel(cfg)` — 完整模型（hybrid 注意力 + Stable LatentMoE）

## attention

- `GatedDeltaNet(d_model, n_heads, head_dim)` — 线性注意力头；`forward(x, chunked=True)`
- `GatedMLA(d_model, n_heads, head_dim)` — 全注意力层
- `KDALayer(...)` — 单层（linear/full 二选一）
- `HybridAttention(cfg)` — 混合栈；`kv_memory(seq_len)` 返回 (full, actual)

## core

- `delta_update(S, k, v, beta)` / `delta_affine(k, v, beta)` / `delta_readout(S, q)`
- `linear_recurrent(keys, values, queries)` / `linear_chunked(...)`
- `scan_sequential(items, combine, identity)` / `scan_chunked(...)`

## moe

- `Router(n_experts, top_k).route(token_id, skill)`
- `gold_experts(token_id, n_experts, top_k)`
- `StableLatentMoE(cfg).forward(x, skill)` / `.load_imbalance(n, skill)`
- `aux_loss(counts, probs, n)` / `imbalance_ratio(counts)`

## mopd

- `Sandbox().make_task(id)` / `task.reward(action)`
- `TeacherEnsemble(n_actions).probs(task)`
- `MOPD(cfg).train(steps)` / `.evaluate(n_tasks)`
- `kl_divergence(p, q)`

## serving

- `SpeculativeDecoder(draft_len).measure(skill)` → (acceptance, speedup)
- `expected_acceptance(a, K)` / `speedup(a, K, c)`
- `BatchScheduler(batch_size).expert_reuse(expert_ids)`
- `KVPrefetcher(n_lookahead).hit_rate(...)`

## CLI

```
python -m kdaforge doctor   # 前向 + 内存口径冒烟
python -m kdaforge memory   # KV 内存对比
python -m kdaforge bench    # 5 项基准
python -m kdaforge train    # MOPD 后训练
python -m kdaforge serve    # HTTP 服务
```
