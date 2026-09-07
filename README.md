# Inpainting Research Agent

一个基于 Hello Agents 的张量分解图像补全研究 Agent：它分析缺失图像、检索方法经验、选择 Matrix/CP/Tucker 基线、提出受约束的 PyTorch 改进，并通过无 Ground Truth 泄漏的公平实验决定接受、拒绝或继续修改候选。

项目目标不是宣称 SOTA，而是展示一条**可验证、可证伪、可复现、可审计**的自主算法研究闭环。

## 30 秒架构

```text
固定研究 Prompt + 图像
          │
          ▼
Hello Agents 编排层
  ├─ Image Profiler ────────────── 只看 corrupted image + mask
  ├─ Knowledge Retriever ───────── 检索本地 Matrix / CP / Tucker 经验
  ├─ Method Selector ───────────── LLM 或确定性 fallback
  ├─ Model Improver ────────────── 输出结构化 idea + 候选模型代码
  └─ Experiment Controller ─────── 最多两轮，记录完整 Trace
          │
          ▼
固定实验内核
  ├─ AST + 独立进程 smoke validation
  ├─ 同预算 paired tuning：M_train 训练，M_val 选优
  ├─ 全部观测像素 final refit
  ├─ 最终 missing PSNR / SSIM
  └─ 固定 Judge + approved algorithm registry
```

核心依赖方向始终是 `Agent → 实验内核`。训练器、指标和 Judge 不依赖 LLM，LLM 也不能修改评估规则或自行宣布成功。

## 为什么使用 Hello Agents

Hello Agents 在这里提供 Tool、ToolRegistry、LLM 适配和 TraceLogger。项目保留它的编排与可观测性能力，但不修改框架源码；所有图像补全算法都位于独立目录中。这样可以展示 Agent 工程能力，同时保持实验内核可独立测试，也方便以后替换模型服务或 Agent 框架。

## Quick start（Linux）

在 `hello_agents` 仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r inpainting_research_agent/requirements.txt

python -m inpainting_research_agent.run \
  --image inpainting_research_agent/assets/example.png \
  --mask-type block \
  --missing-rate 0.4 \
  --max-improvement-rounds 2 \
  --llm-mode off \
  --device auto
```

`--llm-mode off` 使用完全可复现的确定性选择器和 TV 候选模板。要连接 OpenAI-compatible API，可复制 `.env.example` 中的变量并导出到环境，然后使用：

```bash
python -m inpainting_research_agent.run \
  --image inpainting_research_agent/assets/example.png \
  --llm-mode required \
  --device cuda
```

`auto` 会在环境变量完整时调用 LLM，否则安全回退到确定性规则。项目面向 Linux/CUDA 服务器；`--device auto` 会优先选择 CUDA。

## 支持的方法

| 类型 | 方法 | 可学习参数 |
|---|---|---|
| 无训练基线 | Manhattan 最近邻插值 | 无 |
| 张量基线 | Matrix Factorization | 两个矩阵因子与通道偏置 |
| 张量基线 | CP Decomposition | 三个模态因子与通道偏置 |
| 张量基线 | Tucker Decomposition | 核张量、三个因子与通道偏置 |
| 当前 fallback 候选 | Selected decomposition + Total Variation | 保留已选择分解的参数，增加固定形式的 TV 损失 |

所有张量分量均通过 `torch.nn.Parameter` 学习。候选必须继承统一模型接口，不能生成或替换训练流程。

## 无 Ground Truth 泄漏协议

设 `M` 为真实可见像素。调参时再将它确定性划分为：

```text
M = M_train ∪ M_val，M_train ∩ M_val = ∅
```

- 梯度更新只使用 `M_train`；
- 超参数与训练步数只根据 `M_val` 的 MSE 选择；
- 选定配置后重置模型，并在整个 `M` 上重新拟合；
- 人工遮挡区域的 Ground Truth 只在最终评估函数中出现；
- 调参函数的接口中根本不存在 `ground_truth` 参数。

基础模型与候选模型使用相同 trial 数、最大步数、随机种子、数据划分、设备和早停规则。候选只比相应基线多出 idea 引入的超参数。

## 候选代码验证与晋升

候选进入训练前必须依次通过：

1. Pydantic `CandidateProposal` 结构验证；
2. AST 语法、导入、危险调用、父类与接口检查；
3. 独立进程中的构造、forward、有限值、backward、梯度与一步优化检查；
4. manifest 状态和 SHA-256 完整性检查；
5. 与基础张量模型的同预算公平实验。

固定 Judge 的默认准入条件：

```text
missing PSNR delta >= +0.2 dB
composite SSIM delta >= -0.002
trial 数相同，训练过程无 NaN / Inf / 异常
```

通过后，候选及 idea、验证报告、最佳配置、指标、适用条件、来源 run 和代码哈希会进入 `algorithms/approved/<name>/<version>/`。所有已晋升算法共用一个 `AlgorithmRunnerTool`，不会为每个候选复制训练代码。

## 一次真实研究轨迹

固定 Prompt：

> 请分析图像和缺失模式，选择合适的张量分解，并在公平实验下提出、验证和改进一个补全算法。

在 `example.png`、block mask、40% 缺失、seed 42、CPU、每方 4 trials × 200 steps 下：

```text
可见特征分析
→ 本地知识检索选择 Tucker
→ 生成 Tucker + TV 假设
→ AST 与训练 smoke test 通过
→ 基础 Tucker / 候选各执行 4 个配对 trials
→ 候选相对 Tucker：PSNR +0.7230 dB，SSIM +0.0818
→ Judge 接受并晋升候选
→ 全体方法比较后，最终仍输出指标更高的最近邻插值
```

“候选被晋升”和“候选是全体冠军”是两个不同问题。Agent 不会因为自己提出了改进，就隐藏更强的简单基线。

## 真实结果

| 方法 | Missing PSNR ↑ | Composite SSIM ↑ | 最终拟合时间 | 参数量 | 结论 |
|---|---:|---:|---:|---:|---|
| 最近邻插值 | 15.0618 | 0.7271 | N/A | 0 | 全体冠军 |
| 基础 Tucker | 12.7903 | 0.6269 | 0.0985 s | 8,201 | 候选比较基线 |
| Tucker + TV | 13.5133 | 0.7087 | 0.2882 s | 8,201 | 相对 Tucker 通过晋升 |

| Corrupted image | 插值结果 | Tucker + TV |
|---|---|---|
| ![corrupted](docs/demo/corrupted.png) | ![interpolation](docs/demo/interpolation.png) | ![tucker-tv](docs/demo/tucker_tv.png) |

完整机器生成结果见 [`docs/demo/results.json`](docs/demo/results.json)。

## Benchmark

Benchmark 命令支持多图片、多个 mask 和多个缺失率：

```bash
python -m inpainting_research_agent.run_benchmark \
  --images /data/image1.png /data/image2.png /data/image3.png \
  --mask-types random block \
  --missing-rates 0.3 0.5 \
  --device cuda
```

当前仓库只有用户提供的一张合法图片，因此只执行了 `1 image × 2 masks = 2 cases` 的 smoke benchmark：2/2 case 成功，候选晋升 1 次。该结果只验证端到端稳定性，不作为跨图像泛化证据。正式求职展示前，建议补充至少 3～5 张有明确使用权的测试图片。

本次聚合表见 [`docs/demo/benchmark.md`](docs/demo/benchmark.md)。

## 输出与可审计性

每次完整运行至少生成：

```text
outputs/research-agent-<run-id>/
├── state.json
├── report.md
├── best_completion.png
└── traces/
    ├── trace-*.jsonl
    └── trace-*.html
```

Day 4/5/6 子流程分别保存自己的状态、图片、配置、训练曲线、checkpoint 和 Trace；顶层状态只保存结构化摘要与子流程引用。

## 测试

```bash
pytest -q
```

当前共 36 项测试，覆盖 mask、指标、三种张量模型、两阶段训练、Hello Agents Tools、方法选择、候选代码验证、公平配对、Judge、停止条件、统一入口和 benchmark 聚合。

## 已知限制

- 当前真实结果只来自一张图片，不能证明跨数据集优势。
- `composite_ssim` 是项目定义的复合 SSIM，不是标准 masked SSIM。
- AST 与独立进程 smoke test 是学习项目级防护，不是生产安全沙箱。
- 当前 fallback improver 只实现 TV 正则；真实 LLM 虽可输出其他受限 idea，仍可能被验证器拒绝。
- 超参数搜索是最多 5 次的轻量网格抽样，不是成熟的 Bayesian optimization。
- 图像语义分析尚未接入 DepictQA；当前 profiler 只使用可见像素统计特征。
- 当前同步串行执行，未实现 GPU 并行 benchmark 或远程任务队列。

## Future Work

- 接入 vLLM 或 DepictQA，但继续隔离完整图像 Ground Truth；
- 增加 TT、TR、t-SVD 等张量模型和更丰富的受限生命周期 hook；
- 在多数据集、多 mask、多随机种子上建立稳定 benchmark；
- 将完整训练放入带 CPU/GPU、内存和时间限制的容器沙箱；
- 对候选做多案例晋升，避免单图过拟合；
- 接入 MLflow/W&B 与可视化前端；
- 加入人工审批节点和可恢复的异步执行。

## 学习记录

- [Day 1：可信插值基线](DAY1_LEARNING.md)
- [Day 2：张量分解模型与两阶段训练](DAY2_LEARNING.md)
- [Day 3：Hello Agents Tools 与状态机](DAY3_LEARNING.md)
- [Day 4：知识检索与方法选择](DAY4_LEARNING.md)
- [Day 5：候选代码生成与验证](DAY5_LEARNING.md)
- [Day 6：公平实验、反馈迭代与晋升](DAY6_LEARNING.md)
- [Day 7：端到端交付、Benchmark 与求职展示](DAY7_LEARNING.md)

2～3 分钟录屏提纲见 [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md)。
