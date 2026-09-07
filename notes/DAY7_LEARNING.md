# Day 7：端到端交付、Benchmark 与求职展示

## 1. 今天完成什么

前六天已经分别实现了可信数据流程、张量模型、Hello Agents Tool、知识检索、候选生成、代码验证和公平 Judge。Day 7 的目标不是再增加一个算法，而是把这些能力整理成一个别人可以运行、理解和检查的完整项目。

今天新增：

```text
一条统一命令
→ 自动执行 Day 4 方法选择
→ 自动执行 Day 5 候选生成与验证
→ 自动执行 Day 6 公平实验与晋升
→ 汇总所有方法指标
→ 选择真正的全体冠军
→ 保存 best_completion.png
→ 自动生成 report.md
→ 保存顶层状态与 Trace
```

同时增加：

- 多图片 × 多 mask × 多缺失率的 benchmark 驱动器；
- 适合面试官快速阅读的项目 README；
- 精选演示图片和真实指标 JSON；
- `.env.example`；
- 2～3 分钟录屏脚本；
- 统一入口与 benchmark 的自动测试。

主要新增文件：

| 文件 | 作用 |
|---|---|
| `workflow_full.py` | 组合 Day 4/5/6 子工作流 |
| `run.py` | 项目公开的一条命令入口 |
| `reporting.py` | 标准化方法结果并生成 Markdown 报告 |
| `benchmark.py` | 执行 case 网格并聚合均值、标准差、失败数 |
| `run_benchmark.py` | benchmark 命令行入口 |
| `DEMO_SCRIPT.md` | 2～3 分钟求职演示脚本 |
| `docs/demo/` | 精选的真实结果，不包含大型 checkpoint |

## 2. 为什么不把所有代码重写成一个巨大 Workflow

一种直接做法是把 Day 4、Day 5、Day 6 的代码全部复制到 `run.py`。这种方式短期看起来简单，但会产生三个问题：

1. 单独测试某个阶段变得困难；
2. 子阶段异常时，无法知道最后成功到哪里；
3. 同一套候选验证或训练逻辑可能出现多个版本。

现在的顶层工作流采用组合方式：

```python
day4 = run_day4_workflow(...)
day5 = run_day5_workflow(base_run_dir=day4_run_dir, ...)
day6 = run_day6_workflow(
    base_run_dir=day4_run_dir,
    initial_candidate_dir=day5_candidate_dir,
    ...,
)
```

每个子工作流保留独立状态和 Trace，顶层只记录它们的 run ID、完成状态与状态文件路径。这种结构类似生产系统中的父任务与子任务：组合层负责流程，子任务负责单一职责。

## 3. 固定 Prompt 在这里扮演什么角色

默认 Prompt 是：

> 请分析图像和缺失模式，选择合适的张量分解，并在公平实验下提出、验证和改进一个补全算法。

它描述的是研究目标，而不是一段允许任意操作的命令。真正能执行的操作仍由固定 Workflow 和 Tool 白名单决定。

在当前 MVP 中：

- Prompt 启动并标识一次任务；
- Method Selector 收到固定的结构化选择任务、ImageProfile 和检索证据；
- Model Improver 收到固定候选契约、基础代码和实验反馈；
- 用户不能通过 Prompt 让 Agent 改写指标、读取隐藏真值或执行 shell。

这体现了 Agent 工程的重要原则：自然语言描述目标，程序化控制权限和执行边界。

## 4. 顶层状态机

顶层阶段如下：

```text
CREATED
→ METHOD_SELECTION
→ CANDIDATE_GENERATION
→ FAIR_EVALUATION
→ COMPLETED
```

如果 Day 5 候选没有通过代码验证，Workflow 会跳过公平训练，仍然在插值与基础张量模型中选择最佳输出，并生成最终报告。无效候选不会导致基础结果丢失。

如果发生无法恢复的文件、训练或配置错误，顶层状态变为 `FAILED`，并保存：

```json
{
  "last_error": {
    "type": "...",
    "message": "..."
  }
}
```

因此，即使命令失败，也可以从 `state.json` 查看失败阶段，而不是只依赖终端中的最后一行异常。

## 5. 为什么要标准化 `method_results`

不同阶段原本使用不同的结果结构：

- 插值没有参数量和训练时间；
- Day 4 的基础模型指标位于 `results.tensor_metrics`；
- Day 6 的公平基础模型和候选位于 `rounds[-1]`；
- 被拒绝候选有指标，但不能作为最终输出。

`build_method_results()` 将它们统一为：

```python
{
    "algorithm": "tucker_tv_regularized",
    "role": "candidate",
    "metrics": {...},
    "runtime_seconds": 0.2882,
    "parameter_count": 8201,
    "eligible_for_final_output": True,
    "reconstruction": "...png",
}
```

这带来三个好处：

1. 最终冠军选择只需要处理一个统一列表；
2. Markdown 报告和 benchmark 不需要理解每个子工作流的内部结构；
3. 将来添加 TT 或其他候选时，报告层不需要重写。

最终选择只在 `eligible_for_final_output=True` 的方法中进行。未达到晋升门槛的候选仍保留在报告中，但不会成为生产输出。

## 6. 自动报告为什么从状态生成

`reporting.py` 不重新计算实验，也不让 LLM总结指标。它只读取已经落盘的结构化状态并生成 Markdown。

报告包括：

- Run ID 和固定 Prompt；
- 图像与缺失模式；
- 方法选择理由与证据来源；
- 候选研究假设；
- 插值、基础张量和候选指标；
- 公平预算审计；
- Judge 数值依据和停止原因；
- 子流程状态与 Trace；
- 证据边界。

这样可以避免 LLM 在总结阶段把 `+0.0723` 写成 `+0.723`，或者把 reject 描述成 accept。自然语言解释可以交给 LLM润色，但数值表和结论字段应由确定性代码生成。

## 7. 为什么复制一份 `best_completion.png`

原始最佳结果可能位于：

- Day 4 的 `interpolated.png`；
- Day 6 的 `baseline_final/model_completed.png`；
- Day 6 的 `candidate_final/model_completed.png`。

如果用户每次都要进入多层目录寻找结果，统一入口就没有真正完成。因此顶层流程将获胜图片复制为：

```text
outputs/research-agent-<run-id>/best_completion.png
```

同时在状态中保留 `source_reconstruction`，所以仍可追溯它来自哪个算法和子实验。

## 8. Benchmark 设计

Benchmark case 由三个维度组成：

```text
image × mask_type × missing_rate
```

例如：

```text
3 images × 2 masks × 2 missing rates = 12 cases
```

每个 case 都使用独立 seed，并从完整统一入口开始运行。失败不会中止剩余 case，而是记录错误类型、错误信息和耗时。

聚合层按语义角色统计：

- `interpolation_baseline`；
- `tensor_baseline`；
- `candidate`。

每个角色记录：

- Missing PSNR 均值与总体标准差；
- Composite SSIM 均值与总体标准差；
- 平均最终拟合时间；
- 平均参数量；
- 实际完成的 case 数。

整体还记录失败 case 数和候选晋升次数。

为什么按角色而不是始终按算法名聚合？因为不同图片可能选择 Matrix、CP 或 Tucker。按角色可以先回答“Agent 选择的张量基线整体如何”；正式论文级 benchmark 再同时提供按具体算法分组的结果。

## 9. 今天的真实端到端验收

使用你的 `assets/example.png` 执行：

```bash
python -m inpainting_research_agent.run \
  --image inpainting_research_agent/assets/example.png \
  --mask-type block \
  --missing-rate 0.4 \
  --max-improvement-rounds 2 \
  --method-max-steps 200 \
  --fair-max-steps 200 \
  --tuning-trials 4 \
  --llm-mode off \
  --device cpu
```

最终 run：

```text
research-agent-20260906-163255-811438
```

结果：

| 方法 | Missing PSNR | Composite SSIM | 参数量 | 状态 |
|---|---:|---:|---:|---|
| 最近邻插值 | 15.0618 | 0.7271 | 0 | 最终冠军 |
| Tucker | 12.7903 | 0.6269 | 8,201 | 公平基础模型 |
| Tucker + TV | 13.5133 | 0.7087 | 8,201 | 相对 Tucker 晋升 |

统一入口正确生成了：

- 顶层 `state.json`；
- 顶层 `report.md`；
- 顶层 `best_completion.png`；
- Day 4/5/6 子状态；
- 四层 Trace。

## 10. 今天的 smoke benchmark

仓库目前只有一张用户提供且使用权明确的图片，因此没有伪造“5 张图 benchmark”。实际执行：

```text
1 image × [random, block] × 40% missing = 2 cases
```

预算为每个训练阶段 50 steps、每方 2 trials，结果：

| 角色 | Cases | PSNR mean ± std | SSIM mean ± std |
|---|---:|---:|---:|
| 插值基线 | 2 | 17.4134 ± 2.6063 | 0.7864 ± 0.0741 |
| 张量基线 | 2 | 14.9794 ± 2.0983 | 0.6003 ± 0.0201 |
| 候选 | 2 | 15.1965 ± 2.3154 | 0.6233 ± 0.0430 |

2/2 case 完成，候选晋升 1 次。这个结果只用于证明 benchmark driver 和完整流程稳定，不能作为泛化结论。

## 11. 求职展示时重点讲什么

不建议把重点放在“我加了一个 TV loss”。TV 本身并不新颖。更有价值的是下面四个系统设计：

### 11.1 LLM 与确定性工具分工

LLM 负责选择、解释、提出 hypothesis 和生成受约束代码；程序负责数据隔离、训练、指标、Judge 和状态持久化。

### 11.2 候选代码的可信执行链

结构化 schema → AST 策略 → 独立进程 smoke test → 代码哈希 → 固定 Trainer。每道门解决不同风险。

### 11.3 公平、无泄漏的实验协议

基础与候选配对预算一致，隐藏区域只在最终评估中出现。Agent 的成功只能由固定 Judge 宣布。

### 11.4 正确处理失败和证据边界

候选失败时保存反馈并有限迭代；候选晋升后仍与强插值基线比较；单图结果不包装成 SOTA。

这些设计比“调用一次 LLM 然后执行代码”更接近真实 Research Agent 工程。

## 12. 2～3 分钟演示顺序

1. 用 README 架构图说明问题和职责边界。
2. 执行一条 quick start 命令。
3. 打开 `report.md` 展示方法选择证据。
4. 打开候选 `idea.json` 和 `validation.json`。
5. 打开 `paired_configurations.json` 证明公平预算。
6. 展示三张补全图与 Judge 指标。
7. 展示 Trace HTML 和 approved manifest。
8. 以限制与下一步结束。

完整口播稿见 `DEMO_SCRIPT.md`。

## 13. 今天的实践练习

### 练习 1：完整运行一次

在 Linux 服务器使用 `--device cuda` 执行统一入口，找到顶层 report、best image 和三个子状态。

### 练习 2：增加两张合法图片

选择一张纹理丰富图和一张强边缘/规则结构图，运行：

```bash
python -m inpainting_research_agent.run_benchmark \
  --images image1.png image2.png image3.png \
  --mask-types random block \
  --missing-rates 0.3 0.5 \
  --device cuda
```

比较不同条件下 Method Selector 的选择与候选晋升率。

### 练习 3：模拟候选验证失败

让测试生成器输出一个禁止导入 `os` 的候选，确认完整工作流跳过 Day 6，并仍输出插值或基础张量结果。

### 练习 4：准备面试回答

尝试在两分钟内回答：

1. 为什么不用缺失区域 Ground Truth 调参？
2. 为什么不让 LLM生成整个训练脚本？
3. 为什么候选晋升后最终输出仍可能是插值？
4. AST 检查为什么不是安全沙箱？
5. 为什么改进循环必须有最大轮数？

## 14. 一周完成标准

- [x] 固定 Prompt 可以从一条命令启动完整任务。
- [x] Agent 能选择 Matrix/CP/Tucker 并给出本地证据。
- [x] 插值与基础张量模型得到真实 PSNR、SSIM。
- [x] Model Improver 生成结构化 hypothesis 和候选代码。
- [x] 候选经过静态与动态验证。
- [x] 基础与候选经过无泄漏、同预算公平实验。
- [x] Judge 能接受、拒绝并产生下一轮反馈。
- [x] 循环最多两轮并保证停止。
- [x] 成功候选进入带版本的 approved registry。
- [x] 失败候选保留但不能成为最终输出。
- [x] 最终输出图片、指标、配置、代码、报告和 Trace。
- [x] 一张图片上的 random/block smoke benchmark 完成。
- [x] README、真实结果和演示脚本完成。
- [x] 36 项自动测试通过。

## 15. 一周之后怎么继续

最值得优先做的不是继续堆新 Agent，而是补齐实验可信度：

1. 增加 3～5 张合法图片和多个随机种子；
2. 将“单 case 晋升”改为“验证集平均指标晋升”；
3. 接入一个真实 LLM，对生成 hypothesis 的质量做案例分析；
4. 将候选完整训练迁移到有资源限制的 Linux 容器；
5. 再接入 VLM/DepictQA，验证语义特征是否真的改善方法选择；
6. 最后才考虑前端和并行调度。

到这里，这个项目已经具备一个小型 Research Agent 作品的完整骨架。它最有说服力的地方不是候选一定成功，而是每一个结论都能沿着状态、配置、代码哈希、训练曲线、指标和 Trace 追溯。
