# Highspeed serving：投机解码 + 调度 + 预取

K3.1 Pro Highspeed 在权重不变的前提下把解码提速 2-3 倍，靠的是推理工程三件套。

## 投机解码（speculate.py）

小模型（draft）先快速生成 K 个 token，大模型（target）并行验证、接受最长一致前缀。闭式期望接受数（Leviathan 2023）：

    expected = (1 - a^(K+1)) / (1 - a)
    speedup  = expected / (1 + K·c)

a = 接受率，c = draft/target 单 token 成本比。

## 批调度（scheduler.py）

把共享相同 top-k 专家的 token 分组进同一批，让已加载的专家权重被复用，减少访存。

## KV 预取（prefetch.py）

预测接下来几个 token，提前把它们的 KV cache 取进内存，把计算与访存重叠。

## 复现要点

- 投机解码的加速**上界由接受率决定**：a=1 时 `(K+1)/(1+K·c)`，所以「draft 与 target 越一致越快」。
- 接受率 0.5 是随机基线，1.0 是完美 draft，benchmark 里随 skill 单调。
