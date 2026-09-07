# Quick Smoke Benchmark

> 该 benchmark 只有一张图片和两种 mask，只验证端到端稳定性，不支持泛化结论。

## 范围

- 图片：`assets/example.png`
- Masks：`random`、`block`
- 缺失率：40%
- Cases：2
- 成功：2
- 失败：0
- 候选晋升次数：1
- 每方 tuning trials：2
- 每个 trial 最大步数：50

## 聚合结果

| 角色 | Cases | Missing PSNR mean ± std | Composite SSIM mean ± std | 平均最终拟合时间 | 平均参数量 |
|---|---:|---:|---:|---:|---:|
| interpolation baseline | 2 | 17.4134 ± 2.6063 | 0.7864 ± 0.0741 | N/A | 0.0 |
| tensor baseline | 2 | 14.9794 ± 2.0983 | 0.6003 ± 0.0201 | 0.0155 s | 3,591.5 |
| candidate | 2 | 15.1965 ± 2.3154 | 0.6233 ± 0.0430 | 0.0825 s | 5,383.5 |

原始 benchmark run ID：`benchmark-20260906-163316-361018`。
