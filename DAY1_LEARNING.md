# Day 1 学习笔记：建立可信的补全实验基线

## 今天真正要学会什么

Day 1 的重点不是最近邻插值本身，而是建立后续 Research Agent 必须遵守的实验协议：

1. 算法输入与评估 Ground Truth 分离。
2. mask 语义在全项目中保持一致。
3. 随机实验可以通过 seed 复现。
4. 指标只衡量真正需要补全的区域。
5. 每次运行都留下配置、图片、指标和唯一 run ID。

## 1. 为什么从完整图片制造缺失

真实残缺图片通常没有完整 Ground Truth，因此不能计算真实 PSNR/SSIM。研究阶段从完整图片 `X` 出发，人工生成 mask `M`：

```text
M = 1：像素可见
M = 0：像素缺失
Y = M ⊙ X
```

算法只能接收 `Y` 和 `M`。完整图片 `X` 只能在补全完成后进入评估函数。

对应代码：

- `core/data.py::apply_observation_mask`
- `core/pipeline.py::run_day1_baseline`

## 2. 为什么缺失率不足以描述任务

`40% random` 和 `40% block` 的缺失率相同，但难度完全不同：

- 随机缺失的每个洞附近通常都有观测像素。
- 块缺失的中心离已知边界较远，没有直接局部信息。

所以后续 Method Selector 至少需要同时看到缺失率和 mask 类型。Day 4 还会加入孔洞尺度、连通区域、平滑度和通道相关性。

对应代码：`core/masks.py`。

## 3. 最近邻补全为什么使用多源 BFS

最直接的实现是：对每个缺失像素遍历所有观测像素，寻找最近者。它的复杂度接近：

```text
O(缺失像素数 × 观测像素数)
```

当前实现把所有观测像素同时放入队列，用多源 BFS 向缺失区域扩散，复杂度约为：

```text
O(H × W)
```

它寻找的是 Manhattan 距离下的最近观测像素。相同距离的 tie 由确定性队列顺序决定，因此同一输入总会得到相同输出。

对应代码：`core/interpolation.py::nearest_neighbor_fill`。

## 4. 为什么主指标是缺失区域 PSNR

如果 90% 像素本来就是正确的，计算全图 PSNR 会被大量未修改像素稀释。缺失区域 MSE 定义为：

```text
M_missing = 1 - M
MSE_missing = mean((X_hat - X)² | M_missing = 1)
PSNR_missing = 10 log10(1 / MSE_missing)
```

图像取值范围固定为 `[0, 1]`，所以峰值为 1。

对应代码：

- `core/metrics.py::missing_region_mse`
- `core/metrics.py::missing_region_psnr`

## 5. Composite SSIM 是什么

任意形状 mask 没有一个普遍统一的标准 masked SSIM。当前项目先构造：

```text
X_composite = M ⊙ X + (1 - M) ⊙ X_hat
```

再计算 `SSIM(X_composite, X)`。这保证差异只来自补全区域，但已观测区域仍可能稀释平均分，因此必须称为 `composite_ssim`，不能声称它是标准 masked SSIM。

对应代码：`core/metrics.py::composite_ssim`。

## 6. 运行今天的实验

```bash
python3 -m inpainting_research_agent.run_day1 \
  --image inpainting_research_agent/assets/example.png \
  --mask-type block \
  --missing-rate 0.4 \
  --seed 42 \
  --image-size 64
```

运行测试：

```bash
python3 -m pytest \
  --rootdir=inpainting_research_agent \
  inpainting_research_agent/tests \
  -q
```

这里显式设置 pytest root，是因为当前源码目录本身也是 `hello_agents` 包，而系统环境还没有安装 Hello Agents 所需的 Pydantic。这样可以独立验证 core，不会意外加载 Agent 框架。

## 7. 观察输出

每次实验会创建：

```text
outputs/<run_id>/
├── config.json
├── original.png
├── corrupted.png
├── mask.png
├── interpolated.png
└── metrics.json
```

重点检查：

1. `mask.png` 中白色是观测区域、黑色是缺失区域。
2. `corrupted.png` 只在缺失区域被置为配置的 fill value。
3. `interpolated.png` 的观测区域必须与输入一致。
4. `metrics.json` 中记录实际缺失率、seed、指标定义和产物路径。

## 8. 今日练习

### 练习 A：比较 mask 类型

固定图片、缺失率和 seed，分别运行 `random` 与 `block`，记录 PSNR 差异并解释原因。

### 练习 B：验证复现性

使用完全相同参数运行两次。run ID 会不同，但 mask、插值结果和指标应相同。

### 练习 C：验证指标没有被观测区域稀释

阅读 `tests/test_metrics.py::test_missing_region_error_is_not_diluted_by_observed_pixels`，解释为什么只有一个小洞时 missing MSE 仍然等于 1。

### 练习 D：思考 Agent 接口

回答：为什么未来 `InterpolationTool` 应返回文件路径和结构化指标，而不是把整张 NumPy 数组塞进 LLM 上下文？

## 9. 今日完成标准

- 能解释 mask 的语义。
- 能解释 Ground Truth 为什么只能用于最终评估。
- 能区分全图 PSNR 与缺失区域 PSNR。
- 能解释 block mask 通常比 random mask 更难。
- 能用相同 seed 复现实验。
- 能找到一次运行的全部图片、配置和指标。
