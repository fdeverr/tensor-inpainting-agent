# Day 3 学习笔记：把实验内核变成 Agent 工具

## 今天真正要学会什么

Day 3 不接入 LLM。今天的目标是先把一个普通研究程序改造成“可被 Agent 安全调用的研究环境”：

1. 用 Hello Agents 的 `Tool` 定义能力边界。
2. 用 `ToolRegistry` 统一注册和执行工具。
3. 用 `ToolResponse` 返回结构化成功、失败和耗时。
4. 用确定性状态机规定工具的合法执行顺序。
5. 用 `TraceLogger` 保存可审计的完整轨迹。
6. 调参和训练阶段无法读取人工缺失区域 Ground Truth。

核心认识是：**LLM 不是 Agent 的全部**。工具、状态、权限边界、评估协议和执行轨迹共同构成 Agent 系统。先让这些部分可靠，Day 4 再把固定决策替换为 LLM 决策。

## 1. Day 3 的整体结构

```text
example.png
    │
    ▼
AnalyzeImageTool ──► corrupted.png + mask.png + image_profile.json
    │
    ▼
RunInterpolationTool ──► interpolated.png
    │
    ▼
固定选择 Tucker
    │
    ▼
TuneTensorModelTool ──► best hyperparameters + best_step
    │                    只看 held-out observed MSE
    ▼
TrainTensorModelTool ──► 全部 observed pixels 上的最终模型
    │
    ▼
EvaluateReconstructionTool ──► 插值指标 + Tucker 指标
    │                           此时才读取 Ground Truth
    ▼
CompareExperimentsTool ──► winner + promotion_allowed
```

对应代码：

- `agent_tools/research_tools.py`：六个 Tool。
- `agent_tools/registry.py`：ToolRegistry 构造函数。
- `workflow.py`：确定性状态机。
- `run_day3.py`：命令行入口。

## 2. Hello Agents Tool 协议

每个工具继承 Hello Agents 的 `Tool`，至少实现两个方法：

```python
class RunInterpolationTool(Tool):
    def get_parameters(self) -> list[ToolParameter]:
        ...

    def run(self, parameters: dict) -> ToolResponse:
        ...
```

### `get_parameters()` 的作用

它声明工具名、参数类型、说明、是否必需和默认值。Hello Agents 可以据此生成 function-calling schema。以后 LLM 并不是随便拼一段 Bash，而是从这个有限接口中选择动作。

### `ToolResponse` 的作用

工具不返回无法区分成功与失败的自由文本，而是返回：

```text
status      success / partial / error
text        给 LLM 或人阅读的摘要
data        程序继续执行所需的结构化数据
error       稳定错误码和错误消息
stats       time_ms 等运行统计
context     工具名和实际输入参数
```

`ToolRegistry.execute_tool()` 会调用 `run_with_timing()`，因此六个工具无需各自重复计时代码。

## 3. 六个工具分别负责什么

| 工具 | 主要输入 | 主要输出 | 能否读取 Ground Truth |
|---|---|---|---:|
| `analyze_image` | 图片、mask 配置 | corrupted、mask、可见像素画像 | 仅用于生成 mask，不分析隐藏值 |
| `run_interpolation` | corrupted、mask | 插值图片 | 否 |
| `tune_tensor_model` | corrupted、mask、候选配置 | best config、best step | 否 |
| `train_tensor_model` | corrupted、mask、best config | 最终模型、checkpoint | 否 |
| `evaluate_reconstruction` | reconstruction、mask、GT | PSNR、SSIM | 是，仅最终评估 |
| `compare_experiments` | 两个指标 JSON | winner、差值 | 不需要图片 |

每个工具只交换文件路径和小型 JSON，不把完整图片 tensor、模型参数或 checkpoint 放进 LLM 上下文。

### 为什么不让一个工具包办所有事情

如果一个 `run_everything` 工具同时拥有训练数据、Ground Truth 和比较逻辑，那么权限边界只能依赖函数作者保持谨慎。拆开后，`tune_tensor_model` 和 `train_tensor_model` 的函数参数里根本没有 Ground Truth，泄漏更难发生。

工具粒度也不能无限细。例如把每一步梯度更新做成一个 Tool 会产生大量调用开销，并让 LLM 干预本应固定的数值优化过程。这里选择的边界是“一次完整、可验证的研究操作”。

## 4. 调参工具如何工作

Day 3 固定选择 Tucker，但仍然比较两种 Tucker 配置：

```text
trial 0: Tucker (8, 8, 3),  learning rate 0.03
trial 1: Tucker (16,16,3), learning rate 0.03
```

对每个 trial：

1. 使用相同 seed 划分相同的 observed train/validation mask。
2. 只在 train observed pixels 上计算梯度。
3. 只用 validation observed MSE 选择 checkpoint 和 `best_step`。
4. 不计算缺失区域 PSNR/SSIM。

最后选择 validation MSE 最小的 trial。`tuning_result.json` 中显式记录：

```json
{
  "selection_metric": "held_out_observed_mse",
  "ground_truth_used": false
}
```

## 5. 训练工具为何还要重新训练

调参工具产生的是选择证据，不是最终模型。选出配置和 `best_step` 后，`TrainTensorModelTool`：

1. 用相同 seed 重置模型；
2. 合并 train 和 validation；
3. 在 100% observed pixels 上训练 `best_step` 次；
4. 保存最终 checkpoint 和补全图片。

这延续了 Day 2 修正后的两阶段协议。

## 6. 状态机防止非法工具顺序

当前合法状态为：

```text
CREATED
→ ANALYZED
→ INTERPOLATED
→ TUNED
→ TRAINED
→ EVALUATED
→ COMPLETED
```

执行每一步前，`_assert_stage()` 会检查前置状态。执行成功后，`_transition()` 才更新状态并立即写入 `state.json`。

如果调参失败：

```text
stage = INTERPOLATED
last_successful_stage = INTERPOLATED
last_error.tool_name = tune_tensor_model
```

状态不会假装进入 `TUNED`，已经生成的 corrupted image、mask 和插值结果也会保留。这是以后实现重试、恢复和人工介入的基础。

## 7. Trace 记录了什么

Hello Agents 的 `TraceLogger` 同时输出：

- JSONL：程序可以逐行读取、筛选和统计。
- HTML：人可以在浏览器里检查执行过程。

当前 Trace 包含：

- `session_start` / `session_end`；
- `tool_call`；
- `tool_result`；
- `state_transition`；
- `error`。

每次工具调用记录工具名、当前状态、输入配置、输出摘要、错误和耗时。评估专用 Ground Truth 路径不会写入 Trace。

这使你可以回答：Agent 依次调用了什么？调参依据是什么？哪个步骤失败？最后结论来自哪个指标文件？

## 8. 如何运行

在 Linux 服务器进入 Hello Agents 源码目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r inpainting_research_agent/requirements.txt
```

运行完整 Day 3 工作流：

```bash
python3 -m inpainting_research_agent.run_day3 \
  --image inpainting_research_agent/assets/example.png \
  --mask-type block \
  --missing-rate 0.4 \
  --seed 42 \
  --image-size 128 \
  --max-steps 200 \
  --device auto
```

`--device auto` 在有可用 CUDA 的 Linux 服务器上选择 GPU，否则选择 CPU。

## 9. 当前真实运行结果

本次运行 ID：`day3-20260901-111533-89d41d`。

图像与 mask：

```text
shape:             [64, 128, 3]
mask type:         block
missing rate:      0.400024
observed pixels:   4915
missing pixels:    3277
```

调参结果：

| Trial | Tucker rank | Validation MSE | best step | 参数量 |
|---|---|---:|---:|---:|
| 0 | (8, 8, 3) | 0.00895443 | 130 | 1,740 |
| 1 | (16, 16, 3) | 0.00550611 | 130 | 3,852 |

状态机选择 trial 1，并在全部 4915 个观测像素上重新训练 130 步。

最终比较：

| 方法 | Missing PSNR | Composite SSIM |
|---|---:|---:|
| 最近邻插值 | 15.0618 dB | 0.727073 |
| Tucker | 12.6357 dB | 0.628471 |

最终 `winner=baseline`，所以 `promotion_allowed=false`。Agent 工作流没有因为“必须产生新算法”而篡改结论。

## 10. 如何阅读输出目录

```text
outputs/<day3-run-id>/
├── state.json
├── image_profile.json
├── corrupted.png
├── mask.png
├── interpolated.png
├── tuning_result.json
├── interpolation_metrics.json
├── tensor_metrics.json
├── comparison.json
├── tensor_model/
│   ├── model_raw.png
│   ├── model_completed.png
│   ├── model.pt
│   ├── final_fit_history.json
│   └── training_result.json
└── traces/
    ├── trace-*.jsonl
    └── trace-*.html
```

建议按以下顺序阅读：

1. `state.json`：先看全局状态和所有产物索引。
2. `tuning_result.json`：确认选择没有用 Ground Truth。
3. `training_result.json`：确认最终训练用了全部 observed pixels。
4. `comparison.json`：查看可否晋升。
5. `trace-*.jsonl`：还原每一次工具调用。

Linux 上可用 `jq` 查看工具调用顺序：

```bash
jq -r 'select(.event == "tool_call") | .payload.tool_name' \
  inpainting_research_agent/outputs/<day3-run-id>/traces/trace-*.jsonl
```

## 11. 测试覆盖

```bash
python3 -m pytest \
  --rootdir=inpainting_research_agent \
  inpainting_research_agent/tests \
  -q
```

目前共有 21 项测试，Day 3 新增覆盖：

- 六个工具已注册且可以生成 function-calling schema；
- 非法路径返回结构化错误和耗时；
- 完整状态机到达 `COMPLETED`；
- Trace 中工具调用顺序正确；
- 调参和训练结果声明未使用 Ground Truth；
- 故意让调参失败时，状态停留在最后成功的 `INTERPOLATED`。

## 12. 今日练习

### 练习 A：直接调用一个 Tool

绕过状态机，从 `ToolRegistry` 取得 `run_interpolation`，传入 corrupted 和 mask 路径，观察 `ToolResponse.to_dict()` 的结构。

### 练习 B：制造参数错误

把 `missing_rate` 改为 `1.2`，确认返回 `INVALID_PARAM`，并检查 `state.json` 是否仍停在 `CREATED`。

### 练习 C：比较 Tool 和普通函数

对照 `nearest_neighbor_fill()` 与 `RunInterpolationTool.run()`：前者只做算法计算，后者增加了哪些 Agent 工程能力？

### 练习 D：还原轨迹

只阅读 Trace JSONL，不看 `state.json`，写出本次工作流的调用顺序、每步输入和最终 winner。

### 练习 E：扩展候选配置

在 `Day3WorkflowConfig.candidates` 中加入 `(12,12,3)` 和另一个学习率。确认最终选择仍然只依据 validation observed MSE，而不是 missing PSNR。

## 13. 今日完成标准

- 能解释 `Tool`、`ToolRegistry` 和 `ToolResponse` 的职责。
- 能解释为什么数值训练保留在 core，而不是复制到 Tool 中。
- 能画出 Day 3 状态机并说明每个前置条件。
- 能从 Trace 还原完整执行顺序。
- 能说明 Ground Truth 在哪个工具中第一次出现。
- 能解释为什么当前 winner 是插值，而工作流仍然是成功的。
- 能独立运行 Day 3 并找到 state、Trace、checkpoint、补全图和指标。

## 14. Day 4 会改变什么

Day 3 中“选择 Tucker”是写死的：

```text
selected_model = tucker
```

Day 4 将加入本地张量分解经验文档和 Method Selector。LLM 只能根据 `image_profile.json`、插值图分析和检索到的知识片段输出结构化 `MethodPlan`，然后状态机验证该计划是否合法。其余训练、评估和比较工具保持不变。
