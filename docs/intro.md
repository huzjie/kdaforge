# kdaforge 项目介绍（技术篇）

## 今天的热点，背后的技术

2026-10-24 凌晨，月之暗面发布 Kimi K3.1 Pro：2.8T 参数、104B 激活的稀疏 MoE，100 万 token 上下文，Terminal-Bench 4.0 拿 65.7%（压过 GPT-6 Astra）。但真正值得同行学的东西，不在这些数字，而在三个架构决策上：

1. **基座不动，只做后训练**——K3.1 Pro 与 K3 的总参数、专家配置、上下文窗口完全一致，变的全在后训练与推理工程。
2. **1M 上下文靠「混合线性注意力」硬扛**——93 层里 69 层用 KDA（Kimi Delta Attention）循环态替代 KV cache。
3. **后训练不靠闭源教师**——沙箱自动奖励 + 自生成教师（MOPD）。

这三件事，件件都能拆成可复用的工程手法。kdaforge 就是它们的零依赖参考实现。

## 一、混合线性注意力：内存与上下文的解耦

softmax 注意力的 KV cache 是 O(L·d)——1M token 直接把显存打穿。KDA 的做法是：**大部分层用线性注意力，把注意力写成循环状态**。

循环式：`S_t = S_{t-1} - β(S_{t-1} k_t - v_t) k_t^T`（delta 规则，先删后写），状态恒为 d_k×d_v，与 L 无关。

块并行式：这个更新是仿射变换 `S = S·A + b`，两个仿射复合仍是仿射——**可结合**，所以能用关联扫描（associative scan）把整层拆块并行，O(L²) 变 O(L·d)。`core/scan.py` 给了通用幺半群扫描，测试断言 chunked 与 sequential 逐元素相等，零近似。

**可复用结论**：任何「状态更新可写成仿射/幺半群」的序列模型都能块并行，这是 FlashLinearAttention 一类内核的数学根。

## 二、Stable LatentMoE：让 896 个专家别打架

稀疏 MoE 的头号坑是热专家：门控学出偏好后，少数专家吸收大多数 token，成为 serving 瓶颈。两条腿：

- 辅助负载均衡损失 `L = n·Σ f_i P_i`（Switch Transformer 式）
- 逐专家偏置校正 `b_e -= lr·(count_e - mean)/mean`（零成本，只改路由不碰权重）

度量就用 `max_load / mean_load`，1.0 是完美均衡。

## 三、MOPD：后训练不靠闭源教师

沙箱（确定性环境）自动打分，是唯一奖励源；教师是模型自己的更好 rollout（多 seed ensemble）；学生在线采样 → 奖励 → 拉向教师分布。KL 蒸馏与策略梯度方向一致，因为教师本身就擅长高分动作。

**铁律**：训练信号单调趋 1（`skill += lr(1-skill)`），绝不用对错随机游走。

## 四、Highspeed：2-3x 是工程不是魔法

投机解码闭式加速 `speedup = (1-a^(K+1))/(1-a)/(1+K·c)`，加速上限由 draft-target 接受率决定。配专家复用批调度 + KV 预取。

## 怎么用

```bash
python -m kdaforge doctor   # 前向 + 内存口径
python -m kdaforge bench    # memory/retrieval/load_balance/routing/speedup
python -m kdaforge train    # MOPD
python -m kdaforge serve    # HTTP
```

零依赖，Python ≥ 3.8，填好 `configs/*.yaml` 直接跑。

## 完整项目清单

- 16 个 Python 包模块 + 11 个单测文件 + 5 后端 + 5 基准
- 8 个示例 + 9 篇技术文档 + 10 个提示词模板 + 8 个配方
- Docker / K8s / Helm / CI / Makefile / pyproject 全套部署资产
