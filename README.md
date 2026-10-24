# kdaforge

**KDA 混合线性注意力（Gated DeltaNet）+ 896 专家稀疏 MoE 推理引擎 · 零依赖参考实现**

> 拆解月之暗面 Kimi K3.1 Pro（2.8T 参数 / 104B 激活 / 1M 上下文 / Terminal-Bench 4.0 65.7%）的关键架构：纯 Python、零依赖，填好配置直接跑。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://python.org)
![zero-dependency](https://img.shields.io/badge/dependencies-zero-brightgreen)

---

## 一句话讲清技术原理

1M 上下文的大模型，内存头号杀手是注意力层的 KV cache（O(L·d)，随序列线性涨）。**KDA 的思路是：93 层里只留 24 层跑全注意力，另外 69 层换成线性注意力（Gated DeltaNet），用一块恒定大小的循环状态替代 KV cache**。状态更新用 delta 规则（先删后写），并且因为它是仿射变换、可结合，所以能用并行前缀扫描块并行计算——内存从 O(L·d) 降到 O(d_k·d_v)，与上下文长度彻底解耦。

配套三个工程件：**Stable LatentMoE**（896 专家 top-16 的负载均衡，压平热专家）、**MOPD**（沙箱自动奖励 + 自生成多教师的在线蒸馏，不靠闭源教师）、**Highspeed serving**（投机解码 + 批调度 + KV 预取，2-3x）。

## 核心四件套

| 组件 | 解决的问题 | 关键实现 |
|---|---|---|
| `attention/` | 1M 上下文内存 | Gated DeltaNet 循环态 + 关联扫描块并行；Gated MLA 全注意力 |
| `moe/` | 稀疏 MoE 负载均衡 | top-k 门控 + Switch 式 aux loss + 逐专家偏置校正 |
| `mopd/` | 后训练（不靠闭源教师） | 沙箱自动奖励 + 多教师 ensemble + 在线策略蒸馏 |
| `serving/` | 解码提速 2-3x | 投机解码 + 专家复用批调度 + KV 预取 |

## 技术拆解（怎么实现的）

### 1. Delta 规则 = 可并行扫描的仿射变换

Gated DeltaNet 的状态更新：

```
S_t = S_{t-1} - β·(S_{t-1} k_t - v_t) k_t^T
```

它等价于仿射变换 `S_t = S_{t-1} A_t + b_t`，其中 `A_t = I - β k_t k_t^T`、`b_t = β v_t k_t^T`。两个仿射复合仍是仿射：`(A1,b1) ⊕ (A2,b2) = (A1·A2, b1·A2 + b2)`——这就是一个**幺半群**，所以整层能拆块、用前缀扫描 O(L·d_k²) 并行算完，`core/scan.py` 给了通用实现，且 `test_scan.py` 断言 chunked 与 sequential 逐元素相等。

### 2. 负载均衡 = 损失 + 偏置校正

热专家是稀疏 MoE 的吞吐瓶颈。两条腿：辅助损失 `L = n_experts·Σ f_i·P_i` 惩罚集中路由；逐专家偏置 `b_e ← b_e - lr·(count_e - mean)/mean` 在过载时压低该专家。后者只改路由不碰权重，零成本。

### 3. 投机解码 = 接受率决定加速上限

闭式加速 `speedup = (1-a^(K+1))/(1-a) / (1+K·c)`，a 是 draft-target 接受率，c 是成本比。接受率 0.5（随机）到 1.0（完美）之间，加速单调爬升。

## 可直接复用的干货块

**① 线性注意力层的最小实现（30 行）**

```python
from kdaforge.core.delta import delta_update, delta_affine, delta_readout
from kdaforge.core.linalg import matmul, add, identity, zeros
from kdaforge.core.scan import scan_chunked

# 循环式：O(d_k*d_v) 内存
S = zeros(d_v, d_k)
for k, v in zip(keys, values):
    S = delta_update(S, k, v, beta=0.6)   # 先删后写

# 块并行式：幺半群 + 前缀扫描，结果一致
items = [delta_affine(k, v, 0.6) for k, v in zip(keys, values)]
combine = lambda e1, e2: (matmul(e1[0], e2[0]), add(matmul(e1[1], e2[0]), e2[1]))
cum = scan_chunked(items, combine, (identity(d_k), zeros(d_v, d_k)), chunk=16)
```

**② 压平热专家的配方**

```yaml
moe:
  aux_loss_weight: 0.05   # 辅助负载均衡损失
  bias_correction: true   # 逐专家偏置校正
```

**③ 单调训练信号（铁律）**

```python
skill += lr * (1.0 - skill)   # 几何爬升，恒正
# 绝不要用「对 +1 错 -1」的随机游走，低起点会被压到 0
```

## 快速开始

```bash
git clone https://github.com/huzjie/kdaforge
cd kdaforge
python -m kdaforge doctor      # 前向 + 内存口径冒烟
python -m kdaforge bench       # 5 项基准
python -m kdaforge train       # MOPD 后训练
python -m kdaforge serve       # HTTP 服务 (127.0.0.1:8321)
```

无任何第三方依赖，Python ≥ 3.8 即可运行。

## 基准结果（实测）

| 基准 | skill=0.5（随机基线） | skill=1.0（最优） | 说明 |
|---|---|---|---|
| memory（KV 内存比 @1M 上下文） | 3.00x（参考 6 层） | 3.875x（声明 69/24 层） | 结构属性，与 skill 无关 |
| retrieval（长程关联召回） | 0.150 | 1.000 | 单调 |
| load_balance（专家负载不均衡度） | 4.00 | 1.14 | 单调下降 |
| routing（top-k 路由召回） | 0.248 | 1.000 | 单调 |
| speedup（投机解码加速） | 1.044x | 3.684x | 单调（接受率 0.50 → 1.00） |

> 以上为 `python -m kdaforge bench` 实测值：结构属性固定，skill 相关项严格单调。

## 项目结构

```
kdaforge/
  core/          # linalg / delta / linear / scan（纯 Python 内核）
  attention/     # Gated DeltaNet + Gated MLA + 混合栈
  moe/           # Router + Experts + Stable LatentMoE + 负载均衡
  mopd/          # 沙箱 + 教师 + 蒸馏 + MOPD
  serving/       # 投机解码 + 调度 + 预取 + HTTP
  backends/      # mock / cpu / openai / vllm / transformers
  train/ bench/  # 训练 + 5 项基准
  configs/       # default / long_context / balanced / highspeed
tests/           # 11 个单测文件
examples/ docs/ prompts/ recipes/ deploy/
```

详见 [`docs/architecture.md`](docs/architecture.md)、[`docs/linear_attention.md`](docs/linear_attention.md)、[`docs/delta_rule.md`](docs/delta_rule.md)。

## License

MIT — 仅供学习与原型验证，非 Kimi K3.1 官方实现。
