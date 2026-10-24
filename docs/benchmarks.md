# 基准说明

`python -m kdaforge bench` 跑 5 项：

| 基准 | 类型 | 口径 |
|---|---|---|
| memory | 结构（与 skill 无关） | 1M 上下文的 KV 内存，full vs hybrid 的比值 |
| retrieval | 学习（随 skill 单调） | 长程关联召回 @skill 0.5 / 1.0 |
| load_balance | 学习 | 专家负载不均衡度（max/mean）@ 0.5 / 1.0 |
| routing | 学习 | top-k 路由召回 @ 0.5 / 1.0 |
| speedup | 学习 | 投机解码加速 @ 0.5 / 1.0 |

结构项是固定属性；学习项从随机基线单调爬向最优，保证曲线干净可复现。
