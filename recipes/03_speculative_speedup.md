# 配方：投机解码提速

```python
from kdaforge.serving.speculate import SpeculativeDecoder
dec = SpeculativeDecoder(draft_len=8, cost_ratio=0.15)
acc, speedup = dec.measure(skill=1.0)
# speedup = (K+1)/(1+K*c) = 9/2.2 ≈ 4.09x（理想接受率）
```
接受率越低加速越小，实际部署建议先测 draft-target 一致性再定 K。
