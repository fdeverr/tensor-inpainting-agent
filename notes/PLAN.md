# Inpainting Research Agent：7 天可执行开发计划

> 项目目标：基于 Hello Agents 构建一个实验驱动的图像补全研究 Agent。Agent 能够分析图像与缺失模式，从张量分解知识库中选择基础方法，提出受约束的模型改进，通过自动调参和训练评估验证改进，并将通过准入规则的候选算法保存到算法仓库。

## 1. 项目定位

### 1.1 一句话介绍

**Inpainting Research Agent** 是一个面向张量分解图像补全的研究型 Agent：它不是直接“生成一张看起来合理的图片”，而是围绕一个可验证的研究假设，完成方法选择、模型修改、实验运行、指标比较和反馈迭代。

### 1.2 项目要展示的能力

这个项目面向求职展示，重点不是追求 SOTA，而是证明以下能力：

1. 能够设计确定性工具与非确定性 LLM 协作的 Agent 工作流。
2. 能够把算法训练、调参和评估封装为可复用工具。
3. 能够约束和验证 LLM 生成的 PyTorch 代码。
4. 能够设计无 Ground Truth 泄漏、可复现的实验协议。
5. 能够使用客观反馈驱动 Agent 进行有限次数的算法改进。
6. 能够记录完整研究轨迹，并把实验结果包装为可展示项目。

### 1.3 一周版本的完成定义

第 7 天结束时，下面的命令应当可以运行：

```bash
python -m inpainting_research_agent.run \
  --image inpainting_research_agent/assets/example.png \
  --mask-type block \
  --missing-rate 0.4 \
  --max-improvement-rounds 2
```

一次运行至少完成：

```text
读取图片和 mask
→ 分析缺失模式与图像统计特征
→ 运行最近邻插值基线
→ 检索本地张量分解经验文档
→ LLM 在 Matrix / CP / Tucker 中选择基础方法
→ 对基础方法执行相同预算的调参与训练
→ LLM 提出一个仍属于张量分解框架的改进 idea
→ 生成候选模型代码
→ 静态检查与小图训练检查
→ 对候选执行调参与训练
→ 和插值、基础方法进行公平比较
→ 接受候选，或将结构化失败反馈交给下一轮
→ 输出最佳补全图片、指标表、实验轨迹和候选代码
```

### 1.4 一周内明确不做的内容

以下功能统一列入 `Future Work`，不应影响 MVP 交付：

- 不同时接入 vLLM 和 DepictQA；最多选择一个已有可用接口。
- 不实现 TT、TR、t-SVD 等更多张量模型。
- 不允许 LLM 任意生成完整训练脚本或任意 Bash 命令。
- 不构建 Web 前端。
- 不构建多 Agent 协作系统。
- 不做生产级容器沙箱。
- 不做大规模分布式训练和并行超参搜索。
- 不自动修改 Hello Agents 核心源码。
- 不声称 Agent 发明了新的 SOTA 方法。

---

## 2. 技术原则

### 2.1 两层架构

项目必须分成独立实验内核和 Agent 编排层：

```text
inpainting_agent
  ├── 调用 LLM
  ├── 选择工具
  ├── 管理研究状态
  ├── 生成与验证候选
  └── 根据实验反馈迭代
             │
             ▼
inpainting_core
  ├── 数据与 mask
  ├── 张量分解模型
  ├── 训练器
  ├── 调参器
  ├── 指标
  └── 实验产物
```

依赖方向只能是：

```text
inpainting_agent → inpainting_core
```

`inpainting_core` 不允许 import Hello Agents。这样即使将来替换 Agent 框架，实验内核仍然可以独立运行和测试。

### 2.2 LLM 与工具的职责边界

LLM 负责：

- 根据结构化特征和知识文档选择方法。
- 解释选择理由。
- 提出可验证的改进假设。
- 按固定接口生成候选模型代码。
- 阅读结构化实验反馈并提出下一轮修改。

确定性代码负责：

- 图像读取、mask、插值和统计分析。
- 模型训练与超参搜索。
- PSNR、SSIM、时间、参数量计算。
- 候选代码检查。
- 接受或拒绝候选。
- 保存实验产物和状态。

LLM 不得自行宣布算法成功；最终结论只能来自 `ExperimentJudge`。

### 2.3 固定训练器，不生成任意训练流程

所有基础模型和候选模型必须实现统一接口：

```python
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class BaseTensorInpaintingModel(nn.Module, ABC):
    @abstractmethod
    def forward(self) -> torch.Tensor:
        """返回 [H, W, C] 范围在 [0, 1] 的完整重建张量。"""

    def loss_terms(
        self,
        prediction: torch.Tensor,
        observed: torch.Tensor,
        train_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """返回 data_loss 和可选正则项。"""

    @classmethod
    def search_space(cls) -> dict[str, Any]:
        """声明该模型允许搜索的超参数。"""
```

固定 Trainer 只负责调用上述接口。若候选确实需要特殊行为，只允许提供受限的生命周期 Hook；一周 MVP 默认不实现特殊 Hook。

### 2.4 训练、调参与最终评估必须隔离

设完整图像为 `X`，观测 mask 为 `M`，其中 `M=1` 表示观测像素：

```text
Y = M ⊙ X
```

基础训练损失只能使用训练观测像素：

```text
L_obs = ||M_train ⊙ (X_hat - Y)||² / sum(M_train)
```

为了避免调参偷看人工遮挡区域的 Ground Truth，需要从观测像素中再划出少量验证像素：

```text
M = M_train + M_val
M_train ∩ M_val = ∅
```

- 模型拟合只使用 `M_train`。
- 超参选择只使用 `M_val` 上的重建误差。
- 选出最优超参数和训练步数后，重置模型并在 `M_train ∪ M_val` 上完成最终拟合。
- 人工遮挡区域 Ground Truth 只在所有训练和调参完成后，用于最终报告。
- 图像分析和 VLM 只能看到 corrupted image、mask 和插值结果，不能看到完整图。

### 2.5 公平预算

基础模型和候选模型必须采用相同资源预算：

- 相同的超参试验次数，例如各 5 次。
- 相同的最大训练步数。
- 相同的早停规则。
- 相同的图片、mask、随机种子和设备。
- 相同的调参目标。

候选不能因为获得更多训练步骤或更多调参次数而被判定更好。

---

## 3. MVP 范围与默认配置

### 3.1 支持的方法

| 类别 | 方法 | 参数化形式 | 用途 |
|---|---|---|---|
| 插值基线 | 最近邻填充 | 无训练参数 | 最低成本基线 |
| 张量基线 | Matrix Factorization | `U, V` 为 `nn.Parameter` | 最简单低秩基线 |
| 张量基线 | CP | `A, B, C` 为 `nn.Parameter` | 参数高效的三阶分解 |
| 张量基线 | Tucker | 核张量与因子矩阵均为 `nn.Parameter` | 灵活表达空间和通道秩 |

### 3.2 支持的 mask

第一版只支持：

1. `random`：独立随机像素缺失。
2. `block`：一个或多个连续矩形区域缺失。

如果第 7 天仍有时间，再加入 `irregular` 自由形状 mask。

### 3.3 默认实验预算

```yaml
seed: 42
image_size: 128
missing_rates: [0.3, 0.5]
mask_types: [random, block]
max_steps: 1000
early_stopping_patience: 100
tuning_trials: 5
validation_observed_ratio: 0.1
max_improvement_rounds: 2
candidate_timeout_seconds: 300
```

开发时优先使用 `64×64` 或 `128×128` 图片；最终演示再根据设备情况提升到 `256×256`。

### 3.4 候选准入规则

候选必须同时满足：

1. 语法、接口、forward、backward 和一步优化检查全部通过。
2. 在固定验证集上，相比所选基础模型的缺失区域平均 PSNR 提升至少 `0.2 dB`。
3. SSIM 不出现明显下降，默认允许误差为 `0.002`。
4. 所有验证样例均成功完成，没有 NaN、Inf 或超时。
5. 参数量与运行时间被完整记录。

如果一周内候选始终没有超过基础模型，也不应该伪造成功。最终演示可以展示：

```text
Agent 提出假设 → 执行实验 → 发现没有提升 → 给出失败分析 → 拒绝晋升
```

一个能够正确拒绝错误假设的 Research Agent，同样是成功的项目结果。

---

## 4. 建议目录结构

在 Hello Agents 项目中新增独立示例目录：

```text
inpainting_research_agent/
├── README.md
├── run.py
├── config.py
├── schemas.py
├── workflow.py
├── assets/
│   ├── example.png
│   └── validation/
├── configs/
│   ├── quick.yaml
│   └── benchmark.yaml
├── core/
│   ├── __init__.py
│   ├── data.py
│   ├── masks.py
│   ├── interpolation.py
│   ├── metrics.py
│   ├── profiling.py
│   ├── trainer.py
│   ├── tuner.py
│   ├── evaluator.py
│   └── models/
│       ├── __init__.py
│       ├── base.py
│       ├── matrix_factorization.py
│       ├── cp.py
│       └── tucker.py
├── agent/
│   ├── __init__.py
│   ├── prompts.py
│   ├── selector.py
│   ├── improver.py
│   ├── validator.py
│   ├── judge.py
│   └── llm_client.py
├── tools/
│   ├── __init__.py
│   ├── analyze_image.py
│   ├── interpolation.py
│   ├── retrieve_knowledge.py
│   ├── train_model.py
│   ├── tune_model.py
│   ├── evaluate_model.py
│   ├── validate_candidate.py
│   └── compare_results.py
├── knowledge/
│   ├── matrix_factorization.md
│   ├── cp.md
│   ├── tucker.md
│   └── selection_rules.yaml
├── algorithms/
│   ├── candidates/
│   └── approved/
├── outputs/
│   └── .gitkeep
└── tests/
    ├── test_masks.py
    ├── test_metrics.py
    ├── test_models.py
    ├── test_trainer.py
    ├── test_validator.py
    └── test_workflow_smoke.py
```

生成的实验文件不应散落在源码目录，每次运行使用独立目录：

```text
outputs/<run_id>/
├── config.json
├── state.json
├── corrupted.png
├── mask.png
├── interpolated.png
├── baseline/
│   ├── reconstruction.png
│   ├── best_config.json
│   └── metrics.json
├── candidates/
│   └── <candidate_id>/
│       ├── idea.json
│       ├── model.py
│       ├── manifest.json
│       ├── validation.json
│       ├── best_config.json
│       ├── reconstruction.png
│       └── metrics.json
├── comparison.json
├── report.md
└── trace.jsonl
```

---

## 5. 核心数据结构

### 5.1 图像分析结果

```python
class ImageProfile(BaseModel):
    height: int
    width: int
    channels: int
    missing_rate: float
    mask_type: str
    hole_count: int
    largest_hole_ratio: float
    mean_hole_diameter: float
    channel_correlation: float
    local_smoothness: float
    high_frequency_ratio: float
```

第一版不必追求完美特征，只要定义清楚、计算可复现、对方法选择有解释价值。

### 5.2 方法选择结果

```python
class MethodPlan(BaseModel):
    method: Literal["matrix", "cp", "tucker"]
    initial_hyperparameters: dict[str, Any]
    reasons: list[str]
    evidence: list[str]
    confidence: float
```

`evidence` 应引用知识文档中的具体规则，不能只有泛泛的自然语言理由。

### 5.3 实验结果

```python
class ExperimentResult(BaseModel):
    experiment_id: str
    algorithm_name: str
    config: dict[str, Any]
    seed: int
    status: Literal["success", "failed", "timeout"]
    validation_mse: float | None
    missing_psnr: float | None
    composite_ssim: float | None
    runtime_seconds: float
    parameter_count: int
    peak_memory_mb: float | None
    reconstruction_path: str | None
    error: str | None
```

### 5.4 候选 idea

```python
class CandidateProposal(BaseModel):
    candidate_name: str
    base_method: Literal["matrix", "cp", "tucker"]
    hypothesis: str
    proposed_changes: list[str]
    expected_effect: str
    risks: list[str]
    search_space: dict[str, Any]
    code: str
```

### 5.5 研究状态

```python
class ResearchState(BaseModel):
    run_id: str
    status: str
    image_path: str
    mask_path: str | None
    profile: ImageProfile | None
    interpolation_result: ExperimentResult | None
    method_plan: MethodPlan | None
    base_result: ExperimentResult | None
    candidates: list[CandidateProposal]
    candidate_results: list[ExperimentResult]
    feedback_history: list[dict[str, Any]]
    best_algorithm: str | None
    current_round: int
```

推荐的状态枚举：

```text
CREATED
→ PROFILED
→ INTERPOLATED
→ METHOD_SELECTED
→ BASE_TUNED
→ BASE_EVALUATED
→ CANDIDATE_PROPOSED
→ CANDIDATE_VALIDATED
→ CANDIDATE_TUNED
→ CANDIDATE_EVALUATED
→ ACCEPTED / REJECTED
→ COMPLETED / FAILED
```

每次状态变化后立即保存 `state.json`，便于失败恢复和演示 Agent 轨迹。

---

## 6. 工具设计

工具参数尽量传递文件路径、配置和 `run_id`，不要通过 LLM 上下文传递大型 tensor。

| 工具 | 输入 | 输出 | 是否调用 LLM |
|---|---|---|---|
| `analyze_image` | image、mask | `ImageProfile` | 否 |
| `run_interpolation` | observed、mask | 图片与指标 | 否 |
| `retrieve_method_knowledge` | profile、query | 文档片段 | 否 |
| `select_tensor_method` | profile、知识片段 | `MethodPlan` | 是 |
| `tune_tensor_model` | method、search space、预算 | best config | 否 |
| `train_tensor_model` | model、config、mask | 选择阶段 checkpoint、曲线、best step | 否 |
| `fit_tensor_model_on_all_observations` | model、best config、best step、observed mask | 最终 checkpoint、曲线 | 否 |
| `evaluate_reconstruction` | reconstruction、GT | final metrics | 否 |
| `propose_candidate` | base code、profile、metrics、feedback | `CandidateProposal` | 是 |
| `validate_candidate` | candidate package | validation report | 否 |
| `compare_experiments` | base、candidate | judgment | 否 |
| `promote_candidate` | candidate、evidence | approved manifest | 否 |

训练工具内部如需启动独立进程，应使用参数列表并设置 `shell=False`：

```python
subprocess.run(
    [python_executable, "-m", runner_module, "--config", config_path],
    cwd=isolated_workdir,
    timeout=timeout_seconds,
    check=False,
    shell=False,
)
```

Agent 不得直接提交任意 Bash 字符串。

---

## 7. 指标定义

### 7.1 缺失区域 PSNR

最终主指标是人工遮挡区域的 PSNR：

```text
M_missing = 1 - M
MSE_missing = sum(M_missing ⊙ (X_hat - X)²) / sum(M_missing)
PSNR_missing = 10 log10(1 / MSE_missing)
```

图像范围固定为 `[0, 1]`。需要处理缺失区域为空、MSE 为 0 等边界情况。

### 7.2 SSIM 的说明

SSIM 没有天然统一的任意 mask 版本。一周 MVP 推荐报告 `composite SSIM`：

```text
X_composite = M ⊙ X + (1 - M) ⊙ X_hat
SSIM_composite = SSIM(X_composite, X)
```

这意味着已观测区域被原图覆盖，差异只来自补全区域，但大面积观测区域仍可能稀释结果。README 必须明确该定义，不应把它描述成标准 masked SSIM。

### 7.3 其他指标

必须同时记录：

- 调参验证 MSE；
- 训练时间；
- 推理时间；
- 参数量；
- 失败或超时状态；
- 可选的峰值显存。

---

## 8. 7 天详细开发日程

每天学习时间控制在 30～60 分钟，学习内容必须直接服务于当天实现。不要连续看数小时教程后才开始写代码。

### Day 1：建立可信的图像实验协议

#### 当天目标

实现从完整图片生成 corrupted image、运行插值、计算最终指标、保存实验产物的最小闭环。

#### 当天任务

##### 项目骨架与数据协议

1. 创建上文目录结构和必要的 `__init__.py`。
2. 创建 `quick.yaml` 与 `benchmark.yaml`。
3. 定义：
   - `ExperimentConfig`
   - `ExperimentResult`
   - `ResearchState`
4. 实现图片读取：
   - RGB 转换；
   - 缩放到配置尺寸；
   - 转换为 `[H, W, C]` float tensor；
   - 数值范围固定为 `[0, 1]`。
5. 实现随机 mask 和块 mask。
6. 明确 mask 约定：`1=observed`、`0=missing`。
7. 所有随机过程显式接收 seed。

##### 插值与指标

1. 实现最近邻缺失像素填充。
2. 插值只能读取 observed image 和 mask。
3. 实现：
   - `missing_region_mse`
   - `missing_region_psnr`
   - `composite_ssim`
4. 保存：
   - 原始图，仅用于 benchmark；
   - corrupted image；
   - mask；
   - interpolation result；
   - metrics JSON。
5. 为 mask 和指标编写单元测试。

#### 当天验收

运行：

```bash
python -m inpainting_research_agent.core.interpolation \
  --image inpainting_research_agent/assets/example.png \
  --mask-type block \
  --missing-rate 0.4 \
  --seed 42
```

验收清单：

- [ ] 同一个 seed 生成完全相同的 mask。
- [ ] 实际缺失率与目标缺失率误差可解释。
- [ ] 插值过程没有读取缺失区域 Ground Truth。
- [ ] 指标结果为有限数值。
- [ ] 输出目录包含图片、配置和 metrics。
- [ ] `pytest` 中 mask 与指标测试通过。

#### 当天风险与降级方案

- 若 SSIM 依赖安装耗时，先实现 PSNR，当日最后再补 SSIM。
- 若最近邻填充实现复杂，可以先使用固定均值填充打通接口，再替换为最近邻。
- 不允许在 Day 1 提前写 Agent 逻辑。

#### 当天产出

一个完全不依赖 LLM 的可复现实验基线。

---

### Day 2：实现统一 Trainer 和三种张量分解模型

#### 当天目标

实现 Matrix Factorization、CP 和 Tucker，并用同一个 Trainer 完成图像拟合。

#### 当天任务

##### 模型接口与 Matrix Factorization

1. 实现 `BaseTensorInpaintingModel`。
2. 实现 Matrix Factorization：

```text
X_flat ≈ U @ V
U ∈ R^(H×R)
V ∈ R^(R×(W·C))
```

3. `U` 和 `V` 都必须是 `nn.Parameter`。
4. 实现初始化策略并记录初始化 seed。
5. 实现固定 Trainer：
   - Adam；
   - masked observed loss；
   - loss 日志；
   - early stopping；
   - NaN/Inf 检测；
   - checkpoint 与重建图保存。
6. 从 observed mask 中划出 `M_train` 和 `M_val`。
7. 用 `M_val` 选择 `best_step` 后重置模型，并在完整 observed mask 上重新拟合。

##### CP 与 Tucker

1. 实现 CP：

```text
X_hat[i,j,k] = Σ_r A[i,r] B[j,r] C[k,r]
```

2. 实现 Tucker：

```text
X_hat = G ×₁ A ×₂ B ×₃ C
```

3. 核张量和因子矩阵全部使用 `nn.Parameter`。
4. 三个模型统一返回 `[H, W, C]`。
5. 三个模型都实现 `search_space()`。
6. 为每个模型运行 forward、backward 和一步优化测试。

#### 当天验收

分别运行：

```bash
python -m inpainting_research_agent.core.run_experiment --model matrix
python -m inpainting_research_agent.core.run_experiment --model cp
python -m inpainting_research_agent.core.run_experiment --model tucker
```

验收清单：

- [ ] 三个模型均能完成训练。
- [ ] 训练 loss 只使用 `M_train`。
- [ ] `M_val` 只用于早停或超参选择。
- [ ] 最终模型重置后使用 `M_train ∪ M_val` 拟合，不再执行验证早停。
- [ ] 最终缺失区域指标只在训练结束后计算。
- [ ] 模型输出 shape 和输入图一致。
- [ ] 参数全部出现在 `model.parameters()` 中。
- [ ] 训练过程没有 NaN/Inf。
- [ ] 三种模型的实验 JSON Schema 完全一致。

#### 当天风险与降级方案

- 如果 Tucker 实现调试时间过长，先限制通道 rank 为 3。
- 如果高分辨率训练太慢，统一使用 `64×64` 开发图。
- 暂时不加入 TV、稀疏、频域等额外正则，这些留给 Agent 候选。

#### 当天产出

一个稳定的纯 PyTorch 张量补全实验内核。

---

### Day 3：把实验内核封装为 Hello Agents 工具

#### 当天目标

使用 Hello Agents 的 `Tool` 和 `ToolRegistry` 暴露实验能力，并实现一个无 LLM 决策的确定性工作流。

#### 当天任务

##### 工具包装

实现：

1. `AnalyzeImageTool`
2. `RunInterpolationTool`
3. `TrainTensorModelTool`
4. `TuneTensorModelTool`
5. `EvaluateReconstructionTool`
6. `CompareExperimentsTool`

每个工具必须：

- 声明清晰参数；
- 验证路径、范围和枚举值；
- 返回结构化 `ToolResponse`；
- 返回产物路径，不返回完整 tensor；
- 捕获常见错误；
- 记录运行耗时。

##### 确定性工作流

实现第一版 `workflow.py`：

```text
CREATED
→ analyze_image
→ run_interpolation
→ 固定选择 tucker
→ tune_tensor_model
→ train_tensor_model
→ evaluate_reconstruction
→ compare_experiments
→ COMPLETED
```

每一步开始前检查前置状态，结束后保存 `state.json`。

接入 Hello Agents 的 Trace 机制，至少记录：

- 工具名称；
- 输入参数；
- 状态变化；
- 输出摘要；
- 错误；
- 时间。

#### 当天验收

验收清单：

- [ ] 工作流可以一条命令跑通。
- [ ] Tool 中没有复制 Trainer 逻辑，只调用 core。
- [ ] 任一步失败后 `state.json` 保留最后成功阶段。
- [ ] 工具返回中包含 `run_id` 和 artifact path。
- [ ] Trace 可以还原完整执行顺序。
- [ ] 训练超时或参数非法时返回结构化错误。

#### 当天风险与降级方案

- 如果通用 Agent 循环难以控制，直接使用普通 Python 状态机调用 ToolRegistry。
- Day 3 不需要 LLM 自动选择工具；先证明工具层可靠。
- 不要为了复用而修改 Hello Agents 核心 Tool 协议。

#### 当天产出

一个“确定性但已工具化”的研究流水线，为 Day 4 的 LLM 决策准备稳定环境。

---

### Day 4：实现方法知识库和 Method Selector

#### 当天目标

让 LLM 基于图像特征与本地知识文档，在 Matrix、CP 和 Tucker 中做出结构化选择。

#### 当天任务

##### 图像画像和知识文档

完善 `ImageProfile`：

- 图片尺寸；
- 缺失率；
- mask 类型；
- 连通缺失区域数量；
- 最大孔洞占比；
- RGB 通道相关性；
- 局部平滑度；
- 高频能量比例。

编写三份方法经验文档，每份至少包含：

1. 数学形式。
2. 参数量特点。
3. 优势和局限。
4. 对缺失模式的经验判断。
5. 对图像空间结构与通道相关性的经验判断。
6. 推荐 rank 范围。
7. 常见优化失败现象。

编写 `selection_rules.yaml`，把最关键规则结构化，例如：

```yaml
- condition: channel_correlation_high
  prefer: tucker
  reason: Tucker can use a small channel rank while retaining flexible spatial ranks.

- condition: parameter_budget_tight
  prefer: cp
  reason: CP usually uses fewer parameters at the same nominal rank.
```

##### 检索与选择

1. 实现简单本地检索，不需要向量数据库：
   - 读取 Markdown；
   - 按标题切块；
   - 关键词和规则匹配；
   - 返回带来源的片段。
2. 实现 Method Selector Prompt。
3. 强制 LLM 输出 `MethodPlan` JSON。
4. 对输出做 Pydantic 校验。
5. 输出非法时最多修复一次；仍失败则回退到确定性规则。
6. 把选择理由、证据文档和置信度写入 Trace。

固定入口 Prompt 可以是：

```text
请根据当前图像的缺失模式、统计特征、插值结果和张量分解经验文档，
选择一个合适的基础张量分解方法完成图像补全。
```

##### 可选：视觉分析

如果已有稳定 VLM API，可增加 `VisualDescriptionTool`：

- 输入 corrupted image、mask overlay 和 interpolation result；
- 输出纹理强度、主要边缘、重复结构、语义区域；
- 不让 VLM 输出 PSNR/SSIM；
- VLM 失败时不阻塞主流程。

没有现成 API 就跳过，使用确定性图像画像完成 MVP。

#### 当天验收

- [ ] 同一 `ImageProfile` 下选择结果基本稳定。
- [ ] 输出只可能是 matrix、cp 或 tucker。
- [ ] 每个 reason 能关联 profile 特征或文档 evidence。
- [ ] JSON 非法时能修复或回退。
- [ ] 不向 LLM 提供完整 Ground Truth。
- [ ] LLM 选择后，工作流可以自动训练相应基础模型。

#### 当天产出

第一个真正参与研究决策的 Agent 节点。

---

### Day 5：实现候选 idea、代码生成和验证

#### 当天目标

让 LLM 根据基础模型代码和实验结果提出受约束改进，并生成可验证的候选模型。

#### 当天任务

##### 改进空间与 Prompt

允许的改进类型：

- 因子初始化策略；
- rank 的非对称设置；
- 因子正则化；
- Total Variation 正则；
- 因子平滑正则；
- rank 或正则权重调度；
- 多尺度拟合策略，但最终模型仍须以张量分解参数为主体。

禁止的改进：

- 引入预训练扩散模型；
- 使用 CNN/Transformer 替代张量分解主体；
- 读取完整 Ground Truth；
- 调用网络或外部程序；
- 修改评估代码；
- 通过增加不公平训练预算获取提升。

Model Improver 的输入应包含：

- `ImageProfile`；
- Method Selector 的理由；
- 基础模型源代码；
- 基础模型最佳配置；
- 训练曲线摘要；
- 基础指标；
- 上一轮失败反馈；
- 允许和禁止的修改范围；
- 固定模型接口。

输出必须包含：

- hypothesis；
- proposed changes；
- expected effect；
- risks；
- search space；
- 完整候选模型代码。

##### 候选验证

实现两级验证。

第一级：静态检查

- Python AST 能解析；
- import 白名单只允许 `torch`、`torch.nn`、`torch.nn.functional`、`math`；
- 禁止 `os`、`subprocess`、`socket`、`requests`、`shutil`；
- 禁止 `open`、`eval`、`exec`、`compile`、`__import__`；
- 必须包含规定类名；
- 必须继承基础模型接口；
- 必须声明 `search_space()`。

第二级：独立进程 smoke test

1. 动态加载候选模型。
2. 用 `16×16×3` 假数据实例化。
3. 执行 forward。
4. 检查 shape、dtype 和有限值。
5. 计算 masked loss。
6. 执行 backward。
7. 检查至少一个参数获得有限梯度。
8. 执行一步 optimizer update。
9. 设置短超时。

> 注意：AST 白名单和独立进程只是 MVP 安全护栏，不等于生产级安全沙箱。README 必须诚实说明这一限制。

#### 候选目录

每个候选保存为：

```text
algorithms/candidates/<candidate_id>/
├── idea.json
├── model.py
├── manifest.json
└── validation.json
```

`manifest.json` 至少包含：

- candidate ID；
- base method；
- 生成时间；
- LLM 模型；
- prompt version；
- code hash；
- validation status；
- allowed search space。

#### 当天验收

- [ ] 至少一个候选可以通过全部 smoke test。
- [ ] 包含危险 import 的候选会被拒绝。
- [ ] 输出 shape 错误的候选会被拒绝。
- [ ] 没有梯度的候选会被拒绝。
- [ ] 验证失败会生成结构化 feedback。
- [ ] 未通过验证的候选绝不进入正式训练。

#### 当天风险与降级方案

- 如果自由代码生成稳定性差，给 LLM 一个基础模型模板，只允许填写指定区域。
- 如果动态 import 调试耗时，先支持单一固定候选类名。
- 不在 Day 5 实现复杂容器化。

#### 当天产出

一个有明确假设、有代码契约、有安全检查的候选生成环节。

---

### Day 6：自动调参、实验反馈和算法晋升

#### 当天目标

打通“提出候选 → 验证 → 调参 → 评估 → 接受或反馈 → 再改进”的闭环。

#### 当天任务

##### 公平调参

实现 `TuneTensorModelTool`：

1. 读取模型的 `search_space()`。
2. 根据固定 seed 生成最多 5 组配置。
3. 每组配置只使用 `M_train` 训练。
4. 只使用 `M_val` 选择最佳配置。
5. 最终缺失区域 Ground Truth 不参与选择。
6. 保存每次 trial 的配置、状态、验证误差和运行时间。

第一版使用网格搜索或随机搜索即可。只有依赖已经可用时才接入 Optuna。

基础方法和候选方法都要执行相同次数的 trial。

##### 反馈循环

实现最多两轮的研究循环：

```python
for round_index in range(max_improvement_rounds):
    proposal = improver.propose(state)
    validation = validator.validate(proposal)

    if not validation.passed:
        state.feedback_history.append(validation.feedback)
        continue

    best_config = tuner.tune(proposal)
    candidate_result = evaluator.run(proposal, best_config)
    judgment = judge.compare(state.base_result, candidate_result)

    if judgment.accepted:
        registry.promote(proposal, candidate_result, judgment)
        break

    state.feedback_history.append(judgment.feedback)
```

结构化反馈至少包含：

```json
{
  "decision": "rejected",
  "psnr_delta": -0.31,
  "ssim_delta": -0.004,
  "runtime_ratio": 1.2,
  "training_behavior": "validation error decreased early and then diverged",
  "suspected_causes": [
    "TV weight may be too large",
    "spatial rank may be insufficient"
  ],
  "next_round_constraints": [
    "reduce TV search range",
    "do not increase training budget"
  ]
}
```

##### 算法晋升

通过准入后，将候选复制或登记到：

```text
algorithms/approved/<algorithm_name>/<version>/
```

保存：

- 模型代码；
- idea；
- manifest；
- 验证报告；
- 最佳配置；
- 对比指标；
- 适用的 mask 与图像条件；
- 来源 run ID；
- code hash。

MVP 中不要为每个候选动态创建一个新的 Hello Agents Tool 类。使用一个通用的 `AlgorithmRunnerTool`，根据 approved manifest 加载算法，更容易控制和审计。

#### 当天验收

- [ ] 基础模型和候选使用相同 trial 数量。
- [ ] 调参不读取缺失区域 Ground Truth。
- [ ] 第一次失败反馈能进入第二轮 Prompt。
- [ ] 到达最大轮数后一定停止。
- [ ] 接受和拒绝都有明确数值依据。
- [ ] 通过候选可从 approved registry 重新加载运行。
- [ ] 全部失败时系统仍能输出最佳基础模型结果。

#### 当天产出

完整的实验反馈驱动自主改进闭环。

---

### Day 7：Benchmark、README、演示与求职包装

#### 当天目标

稳定端到端流程，完成可复现 benchmark，并把技术决策和实验轨迹包装成容易理解的求职项目。

#### 当天任务

##### 端到端回归与 Benchmark

准备 5～10 张有合法使用权的测试图片。至少覆盖：

- 平滑区域较多的图片；
- 纹理丰富的图片；
- 强边缘与规则结构图片。

推荐最终 benchmark：

```text
5 张图片
× 2 种 mask
× 2 个缺失率
= 20 个 case
```

如果运行时间不足，降级为：

```text
3 张图片 × 2 种 mask = 6 个 case
```

记录：

- 每种方法的均值和标准差；
- 失败数；
- 平均运行时间；
- 平均参数量；
- 每个 case 的完整配置。

执行端到端 smoke test，确保从空输出目录开始能完成一次 quick run。

##### README 与结果报告

README 必须包含：

1. 项目一句话介绍。
2. 30 秒能看懂的架构图。
3. 为什么 LLM 只负责推理，工具负责实验。
4. 为什么使用 Hello Agents。
5. 安装与 quick start。
6. 支持的张量分解方法。
7. 无 Ground Truth 泄漏的实验协议。
8. 候选代码验证流程。
9. 一次完整研究轨迹。
10. 结果表和补全图片。
11. 已知限制。
12. Future Work。

推荐结果表：

| 方法 | Missing PSNR ↑ | Composite SSIM ↑ | 时间 ↓ | 参数量 |
|---|---:|---:|---:|---:|
| 最近邻插值 | 21.30 | 0.712 | 0.1 s | 0 |
| 基础 Tucker | 25.81 | 0.842 | 18 s | 25 K |
| Agent Candidate | 26.34 | 0.856 | 22 s | 25 K |

表中数值必须来自真实运行，不能把示例数值直接用于最终 README。

##### 演示视频与项目清理

录制 2～3 分钟演示：

```text
0:00–0:20  问题和项目目标
0:20–0:45  一条命令启动 Agent
0:45–1:20  展示方法选择理由和知识证据
1:20–1:50  展示候选 idea、验证和实验反馈
1:50–2:20  展示最终补全图片与指标表
2:20–2:40  总结架构决策与未来工作
```

清理检查：

- API Key 只通过环境变量读取；
- 提交 `.env.example`，不提交 `.env`；
- 不提交大型 checkpoint；
- 输出目录只保留精选演示结果；
- 所有路径支持相对项目根目录解析；
- 依赖列表完整；
- quick run 命令从干净环境可执行；
- 测试全部通过；
- 如果使用 Git，创建 `v0.1.0` tag。

#### 最终验收

- [ ] 固定 Prompt 可以启动完整任务。
- [ ] Agent 能选择基础张量分解并给出证据。
- [ ] 插值和基础模型都有真实指标。
- [ ] LLM 至少提出一个结构化研究假设。
- [ ] 候选代码经过静态和动态验证。
- [ ] 候选经过公平调参与最终评估。
- [ ] Agent 能基于失败反馈完成第二轮或正确停止。
- [ ] 优胜候选被登记，失败候选被保留但不晋升。
- [ ] 最终输出补全图、PSNR、SSIM、配置、代码和 Trace。
- [ ] README 能在 3 分钟内让面试官理解项目价值。

---

## 9. 每日进度记录模板

每天结束后，在 `devlog/day-N.md` 或项目 Issue 中记录：

````markdown
# Day N

## 今天完成

- ...

## 验收命令

```bash
...
```

## 实际结果

- Tests: ...
- Runtime: ...
- Key metrics: ...

## 遇到的问题

- 问题：...
- 根因：...
- 处理：...

## 技术决策

- 决策：...
- 原因：...
- 代价：...

## 明天第一件事

- ...
````

每日结束条件：

1. 当天新增核心路径至少有一个自动测试。
2. quick run 仍然可运行。
3. 没有未记录的关键技术债务。
4. 把次日第一项任务写得足够具体，第二天可以直接开始。

---

## 10. 测试策略

### 10.1 单元测试

必须覆盖：

- mask 缺失率和 seed 可复现；
- metrics 边界情况；
- 三种模型 forward shape；
- 三种模型 backward 有梯度；
- Trainer 不读取最终缺失区域；
- Pydantic Schema 拒绝非法 LLM 输出；
- Validator 拒绝危险代码；
- Judge 接受和拒绝规则。

### 10.2 集成测试

至少包含：

1. `32×32` 图片的无 LLM 快速工作流。
2. 使用 mock LLM 返回固定 `MethodPlan`。
3. 使用 mock LLM 返回固定合法候选代码。
4. 候选验证失败后进入反馈分支。
5. 达到最大改进轮数后停止。

CI 中优先运行 mock LLM，避免依赖真实 API 和产生费用。

### 10.3 端到端测试

真实 LLM 端到端测试可以手动触发，不必在每次测试中运行。固定保存一条成功 Trace 作为演示，但 README 需要注明 LLM 输出可能存在随机性。

---

## 11. Prompt 与版本管理

### 11.1 Method Selector 约束

System Prompt 应包含：

- 只能选择允许的方法；
- 必须引用提供的 evidence；
- 不得假设看到了完整原图；
- 不得编造指标；
- 必须输出 JSON Schema。

### 11.2 Model Improver 约束

System Prompt 应包含：

- 目标是提出可证伪假设，而不是保证提升；
- 方法主体必须保持在张量分解框架内；
- 不得修改 Trainer、Evaluator 和 Judge；
- 不得读取文件、访问网络或启动进程；
- 不得增加实验预算；
- 必须遵守固定模型接口；
- 必须解释每项改动与实验反馈的对应关系。

### 11.3 版本记录

每次 LLM 调用记录：

- provider；
- model；
- temperature；
- prompt version；
- input artifact hashes；
- raw output path；
- parsed output；
- token usage；
- latency。

这样可以解释同一实验为什么出现不同候选，也方便以后做 Agent Evals。

---

## 12. 风险清单与处理优先级

| 风险 | 表现 | 处理方式 |
|---|---|---|
| 训练太慢 | 一次候选耗时过长 | 降低图片尺寸、steps 和 trial 数 |
| LLM JSON 不稳定 | Schema 校验失败 | 强制结构化输出、一次修复、规则回退 |
| 候选代码经常报错 | 无法进入实验 | 使用模板填空、缩小允许修改范围 |
| 候选没有提升 | 无法“晋升” | 展示正确拒绝与失败分析，不伪造结果 |
| 指标泄漏 | 调参利用 Ground Truth | 观测像素 train/val 划分，最终指标后算 |
| 比较不公平 | 候选预算更大 | Judge 检查 trial 数和训练预算 |
| API 不稳定 | VLM/LLM 阻塞流程 | 缓存响应、提供 mock、VLM 设为可选 |
| 动态代码危险 | 访问系统资源 | AST 白名单、独立进程、超时；说明限制 |
| 工程范围失控 | 一周无法交付 | 严格执行 Out of Scope 列表 |

优先级顺序：

```text
实验正确性
> 端到端可运行
> 失败可恢复
> Agent 决策质量
> 视觉效果
> 更多算法和界面
```

---

## 13. 求职展示重点

### 13.1 面试时的项目叙述

可以按以下顺序介绍：

1. 传统 Agent 容易把自然语言推理和实验执行混在一起。
2. 本项目将 LLM 限制在方法选择和研究假设生成阶段。
3. 训练、调参、指标和准入均由确定性工具完成。
4. 为防止算法“通过偷看答案变好”，调参只使用 held-out observed pixels。
5. LLM 生成代码必须通过 AST 和小图训练检查。
6. 候选是否晋升由固定实验规则决定，而非 LLM 自评。
7. 全部轨迹可以复现和审计。

### 13.2 README 中最值得展示的研究轨迹

理想示例：

```text
图像画像：40% 块状缺失，通道相关性高，局部较平滑
→ Agent 检索 Tucker 文档并选择 Tucker
→ 基础 Tucker 完成调参与评估
→ Agent 提出 TV-Regularized Tucker 假设
→ 第一轮 TV 权重过大，validation error 后期上升
→ Judge 拒绝，并反馈过度平滑与正则范围问题
→ 第二轮缩小 TV 权重范围并调整空间 rank
→ 候选在固定验证集达到准入阈值
→ 候选登记到 approved registry
```

如果真实结果没有提升，则展示真实失败轨迹：

```text
候选两轮均未超过基础模型
→ Agent 正确停止
→ 返回基础 Tucker 作为最佳算法
→ 报告失败原因和下一步研究建议
```

### 13.3 简历描述参考

完成后根据真实结果改写为类似内容：

```text
Built an experiment-driven research agent on Hello Agents for tensor-decomposition
image inpainting, combining structured method selection, constrained PyTorch code
generation, automated hyperparameter tuning, reproducible evaluation, and
metric-gated candidate promotion.
```

不要在没有数据支持时写“显著提升”“自动发明新算法”或“SOTA”。

---

## 14. 最终交付物

项目完成时应具有：

### 代码

- 独立的 `inpainting_core`。
- Hello Agents 工具和研究工作流。
- Matrix、CP、Tucker 三种模型。
- 候选代码生成与验证器。
- 调参、训练、评估和 Judge。
- 候选算法仓库。
- 自动测试。

### 实验产物

- 至少一次成功端到端 Trace。
- 插值、基础方法和候选方法结果。
- 指标 JSON 和对比表。
- 原图、corrupted image、mask 和补全图片。
- 一份自动生成的 `report.md`。

### 求职材料

- 完整 README。
- 架构图。
- 一条真实研究轨迹。
- 2～3 分钟演示视频。
- 一条可复制的 quick start 命令。
- 清晰的 Limitations 与 Future Work。

---

## 15. 一周执行总检查表

### Day 1

- [ ] 数据、mask、插值、指标可运行。
- [ ] 实验结果可保存和复现。

### Day 2

- [ ] Matrix、CP、Tucker 使用统一接口训练。
- [ ] train/validation/final evaluation 隔离。
- [ ] 选择结束后使用全部观测像素完成 final refit。

### Day 3

- [ ] 实验模块封装为 Hello Agents 工具。
- [ ] 确定性状态机和 Trace 可运行。

### Day 4

- [ ] 本地知识库完成。
- [ ] Method Selector 输出合法 `MethodPlan`。

### Day 5

- [ ] LLM 能生成结构化 idea 和候选模型。
- [ ] 候选经过静态和动态验证。

### Day 6

- [ ] 基础与候选公平调参。
- [ ] 反馈迭代、停止和晋升逻辑完成。

### Day 7

- [ ] Benchmark、README、演示和测试完成。
- [ ] 项目可以作为求职作品公开展示。

---

## 16. 项目成功标准

本项目的成功不要求候选算法必然超过基础模型。真正的成功标准是：

> Agent 能形成一个有依据、可证伪的张量分解改进假设；生成符合约束的实现；在不泄漏 Ground Truth、预算公平且可复现的条件下完成实验；根据客观结果接受、拒绝或继续改进该假设；最后输出完整、可审计的研究轨迹。

只要这条闭环稳定运行，这就是一个完整且有技术含量的 Research Agent 求职项目。
