# Day 2 学习笔记：统一张量模型与训练协议

## 今天真正要学会什么

Day 2 的重点不是比较哪种分解最好，而是建立后续 Agent 可以安全调用的统一训练接口：

1. Matrix、CP、Tucker 都只暴露同一种 `forward()` 和 `loss_terms()` 接口。
2. 所有分解因子与 Tucker 核心都由 `nn.Parameter` 直接学习。
3. Trainer 不知道模型内部使用哪种分解。
4. 人工缺失区域不参与训练、早停或超参数选择。
5. 先用 train/validation 选择训练步数，再重置模型并用全部观测像素完成最终拟合。
6. 训练失败、指标较差同样要被忠实记录，而不是由 LLM 修饰结论。

## 1. 三种基础模型

### Matrix Factorization

将 RGB 图片从 `[H, W, C]` 展开为 `[H, W·C]`：

```text
X_flat ≈ U V
U ∈ R^(H×R)
V ∈ R^(R×WC)
```

学习参数：

- `left_factor`
- `right_factor`
- `channel_bias`

对应代码：`core/models/matrix_factorization.py`。

### CP Decomposition

将图片表示为多个秩一张量之和：

```text
X[i,j,k] ≈ Σ_r A[i,r] B[j,r] C[k,r]
```

学习参数：

- `height_factor`
- `width_factor`
- `channel_factor`
- `channel_bias`

对应代码：`core/models/cp.py`。

### Tucker Decomposition

使用核心张量连接三个维度的因子：

```text
X ≈ G ×₁ A ×₂ B ×₃ C
```

学习参数：

- `core`
- `height_factor`
- `width_factor`
- `channel_factor`
- `channel_bias`

Tucker 可以分别设置空间秩和通道秩。对应代码：`core/models/tucker.py`。

## 2. 为什么加入 channel bias

CP 和 Tucker 的初始输出由多个随机因子相乘得到。如果所有因子都很小，输出和梯度也可能很小。

当前模型使用可用于本阶段训练的观测像素 RGB 均值初始化一个可学习 `channel_bias`，分解因子主要学习相对于通道均值的结构残差：

- 选择阶段只使用 `train_mask`，避免读取 validation。
- 最终拟合阶段使用完整 `observed_mask`。
- 两个阶段都不会读取人工缺失区域。

偏置没有替代低秩主体；它相当于一个简单均值项，并且同样是 `nn.Parameter`。

## 3. selection、final refit 和 final test 的隔离

原始 observation mask 先表示算法真正可见的位置：

```text
M_observed = M_train ∪ M_validation
M_train ∩ M_validation = ∅
```

选择阶段中，三个区域的职责：

| 区域 | 是否用于梯度 | 是否用于早停 | 是否用于最终指标 |
|---|---:|---:|---:|
| train observed | 是 | 否 | 否 |
| validation observed | 否 | 是 | 否 |
| artificial missing | 否 | 否 | 是 |

validation 用于选择 `best_step`。选择完成后，丢弃这一阶段的模型权重，用相同 seed 重置模型，并训练恰好 `best_step` 次：

```text
M_final_fit = M_observed = M_train ∪ M_validation
```

因此最终交付的模型确实利用了 100% 的可见像素。人工缺失区域 Ground Truth 仍然只在全部训练完成后计算 PSNR/SSIM，既不会参与梯度，也不会影响训练步数。

对应代码：

- `core/masks.py::split_observed_mask`
- `core/trainer.py::train_tensor_model`：选择阶段
- `core/trainer.py::fit_tensor_model_on_all_observations`：最终拟合阶段

## 4. Trainer 为什么不接收完整原图

两个训练函数都只接收以下几类参数：

```text
model_name
model_hyperparameters
observed_image
observed_mask
training_config
seed
```

没有 `ground_truth` 参数。这比在 Prompt 中写一句“不要偷看 Ground Truth”更可靠，因为接口本身让 Trainer 无法访问它。

完整原图只存在于 `day2_pipeline.py` 的最终 `_evaluate()` 阶段。

## 5. 固定 Trainer 做了什么

完整训练流程分为两个阶段：

```text
设置随机种子
→ 解析 auto/cpu/cuda 设备
→ 划分 train/validation mask
→ 仅用 train pixels 初始化通道均值
→ 创建模型
→ masked observed loss
→ backward + Adam
→ 定期计算 validation MSE
→ 保存最佳 checkpoint
→ early stopping
→ 得到 best_step
→ 用相同 seed 重置模型
→ 使用全部 observed pixels 重新拟合 best_step 次
→ 保存最终 checkpoint
→ 输出 clamp 到 [0,1] 的最终重建
```

它还会检查 loss 和梯度是否出现 NaN/Inf。

Linux 服务器上 `--device auto` 会在 `torch.cuda.is_available()` 为真时自动选择 CUDA。

这种做法把“选择训练多久”和“充分利用所有已知信息”分开。最终拟合不再看 validation loss，也不会再次早停，否则又相当于保留了一部分观测像素不用。

## 6. raw reconstruction 与 completed reconstruction

最终拟合模型会为整张图片生成预测，因此输出两张图：

- `model_raw.png`：模型对所有像素的原始预测。
- `model_completed.png`：观测位置复制原始观测值，只在洞内使用模型预测。

最终补全结果应使用 `model_completed.png`，因为 inpainting 不应该修改已知像素。

## 7. 运行三个模型

先在 Linux 服务器创建环境，并根据服务器 CUDA 驱动选择合适的 PyTorch 安装方式。然后安装项目依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r inpainting_research_agent/requirements.txt
```

Matrix：

```bash
python3 -m inpainting_research_agent.run_day2 \
  --image inpainting_research_agent/assets/example.png \
  --model matrix \
  --rank 16 \
  --mask-type block \
  --missing-rate 0.4 \
  --device auto
```

CP：

```bash
python3 -m inpainting_research_agent.run_day2 \
  --image inpainting_research_agent/assets/example.png \
  --model cp \
  --rank 12 \
  --mask-type block \
  --missing-rate 0.4 \
  --device auto
```

Tucker：

```bash
python3 -m inpainting_research_agent.run_day2 \
  --image inpainting_research_agent/assets/example.png \
  --model tucker \
  --rank-h 16 \
  --rank-w 16 \
  --rank-c 3 \
  --mask-type block \
  --missing-rate 0.4 \
  --device auto
```

## 8. 当前真实实验结果

在 `128×64`、40% block mask、seed 42、最多 200 selection steps 的相同预算下：

| 方法 | Selection PSNR | Final-refit PSNR | Final SSIM | 参数量 | best step |
|---|---:|---:|---:|---:|---:|
| 最近邻插值 | - | 15.0618 dB | 0.727073 | 0 | - |
| Matrix rank 16 | 12.8820 dB | 12.9952 dB | 0.624284 | 7,171 | 160 |
| CP rank 12 | 12.3293 dB | 11.9284 dB | 0.629802 | 2,343 | 200 |
| Tucker (16,16,3) | 12.7474 dB | 12.6358 dB | 0.628503 | 3,852 | 130 |

`Selection PSNR` 仅用于诊断，来自只训练 `M_train` 的模型；真正对外报告的是 `Final-refit PSNR`。这些只是固定超参数的基础结果，不是调参后的算法排名。

这里还有一个很重要的实验现象：使用全部观测像素后，Matrix 的洞内 PSNR 上升了，但 CP 和 Tucker 反而下降了。使用全部观测点是正确的最终拟合协议，却不保证隐藏区域指标单调上升。原因可能包括优化轨迹改变、`best_step` 对最终拟合并非最优，以及随机 validation 像素与连续块状缺失的分布不一致。这正是后续 Agent 应形成并验证的研究假设，而不是修改评估结果的理由。

## 9. 为什么张量模型暂时低于插值

模型在随机划出的 validation observed pixels 上误差不高，但目标洞是一个大块连续区域。两者存在分布差异：

- validation 像素周围有大量训练观测。
- 大块洞中心远离任何观测像素。
- 基础分解没有显式空间平滑或边缘连续性约束。
- 洞内参数只受到全局低秩结构间接约束，因此出现条纹。

这揭示了后续 Agent 可以研究的两个问题：

1. 是否应该使用与目标 mask 形态更相似的验证 mask？
2. 是否应在 Tucker/CP 中加入 TV 或因子平滑正则？

当前不能因为结果不好就增加测试信息或修改指标。Research Agent 的价值正是记录失败，并形成下一步可验证假设。

## 10. 运行测试

```bash
python3 -m pytest \
  --rootdir=inpainting_research_agent \
  inpainting_research_agent/tests \
  -q
```

测试覆盖：

- train/validation mask 严格互斥且不包含人工缺失区域；
- 最终拟合使用全部 observed mask，且不接收 Ground Truth；
- 三种模型的 forward shape；
- 所有分解参数均有有限梯度；
- 固定 seed 的训练可复现；
- Day 2 流水线保存 checkpoint、mask、曲线、图片和指标。

## 11. 今日练习

### 练习 A：理解接口隔离

阅读 `train_tensor_model()` 的函数签名，确认它为什么不可能直接读取人工缺失区域 Ground Truth。

### 练习 B：比较 random 与 block

固定模型、rank、seed 和训练预算，只修改 `--mask-type`，观察 random mask 是否更容易被低秩模型补全。

### 练习 C：修改 rank

对 Matrix 分别尝试 rank 4、8、16、32，记录 validation MSE 与最终 missing PSNR。观察二者是否总是同步变化。

### 练习 D：阅读训练曲线

打开 `selection_history.json`，找出 validation MSE 最低的 step；再打开 `final_fit_history.json`，确认最终拟合恰好运行了对应步数，并且没有 validation 指标。

### 练习 E：解释失败图像

查看三种 `model_completed.png` 的洞内条纹，分别从模型表达能力、mask 分布和缺少空间先验三个角度解释。

## 12. 今日完成标准

- 能写出 Matrix、CP、Tucker 的重建公式。
- 能解释 `nn.Parameter` 在三个模型中对应什么。
- 能解释为什么 Trainer 不应该知道模型内部结构。
- 能区分 train observed、validation observed 和 artificial missing。
- 能解释为什么模型选择阶段要留 validation，而最终拟合阶段要使用全部 observed pixels。
- 能解释当前基础模型为什么没有超过插值。
- 能运行至少一种模型并找到 checkpoint、训练曲线和最终指标。
