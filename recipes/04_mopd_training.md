# 配方：MOPD 后训练

```python
from kdaforge.mopd.mopd import MOPD
mopd = MOPD(cfg.mopd)
mopd.train(cfg.mopd.max_steps)   # 在线采样 + 沙箱奖励 + 多教师蒸馏
acc = mopd.evaluate(200)          # 成功率从 ~0.125 爬到接近 1.0
```
关键：skill 更新用 `lr*(1-skill)` 单调爬升，绝不用对错随机游走。
