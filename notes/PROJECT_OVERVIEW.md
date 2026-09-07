# Inpainting Research Agent 项目总览（Day 1–7）

> **一句话**：一个「张量分解图像补全」的 Research Agent——从方法选择、候选改进、公平实验到版本化晋升的完整闭环，且每个结论都能沿 `状态 / 配置 / 代码哈希 / 指标 / Trace` 追溯。

---

## 一、七天能力演进（核心主线）

| Day | 主题 | 核心能力 | 关键文件 |
|---|---|---|---|
| 1 | 可信实验基线 | GT 分离、mask 语义统一、seed 复现、只评缺失区、唯一 run ID | `run_day1.py`、`core/{data,masks,metrics,interpolation}.py` |
| 2 | 张量模型与训练 | Matrix/CP/Tucker 统一 `forward/loss_terms`；train/val 选步数 → 全量重训 | `core/models/*`、`core/trainer.py` |
| 3 | 变成 Agent 工具 | Tool 边界、Registry、ToolResponse、确定性状态机、TraceLogger | `agent_tools/*`、`workflow.py` |
| 4 | 检索增强方法选择 | ImageProfile、本地知识库、Pydantic MethodPlan、LLM+规则 fallback | `workflow_day4.py`、`method_selector.py`、`knowledge/` |
| 5 | 候选生成与验证 | schema 约束、LLM/确定性模板、AST 静态 + 独立进程 smoke | `candidate/{schemas,generator,validator,smoke_runner}.py` |
| 6 | 公平实验与晋升 | 成对配置、GT 隔离、确定性 Judge、晋升/改进闭环 | `workflow_day6.py`、`core/{fair_experiment,experiment_judge}.py` |
| 7 | 端到端交付 | 组合子工作流、统一入口、标准化报告、benchmark | `workflow_full.py`、`run.py`、`reporting.py`、`benchmark.py` |

**演进逻辑**：先建立可信的数据/评估协议（1–2）→ 封装成 Agent 可调用的工具（3）→ 逐步引入 LLM 决策（4 选方法、5 造候选、6 验候选）→ 组装成可交付项目（7）。

---

## 二、完整架构（一条命令的数据流）

```text
run.py（Day 7 顶层 FullResearchWorkflow）
  │
  ├─ 阶段1 方法选择（Day 4）
  │     analyze_image：完整图 → 自动造 mask/corrupted，GT 藏到 evaluation_ground_truth.png
  │     run_interpolation：最近邻插值基线
  │     knowledge/ 检索 + MethodSelector（LLM/规则）→ MethodPlan → 选 Matrix/CP/Tucker
  │     train：训练基础张量模型
  │
  ├─ 阶段2 候选生成（Day 5）
  │     load_improver_context：读 Day4 上下文（不给 GT）
  │     CandidateGenerator（LLM / 确定性 TV 模板）→ CandidateProposal
  │     CandidateValidator：AST 静态检查 → 独立进程 smoke test
  │
  ├─ 阶段3 公平实验（Day 6，候选验证通过才跑）
  │     paired_trial_configurations：基线与候选成对配置（共享参数一致）
  │     tune：GT 隔离调参 → final_fit_and_evaluate：GT 首次进入，算 PSNR/SSIM
  │     judge_candidate：确定性门禁 → 接受则晋升 / 拒绝则带反馈改进（≤2 轮）
  │
  └─ 汇总
        build_method_results：标准化所有方法
        → 在 eligible 方法中选 Missing PSNR 最高者
        → 复制 best_completion.png + 生成 report.md
```

---

## 三、目录结构

```text
inpainting_research_agent/
├── core/               # 无 LLM 的可信内核
│   ├── data.py         #   图片读写、mask 应用
│   ├── masks.py        #   自动生成 mask（random/block）
│   ├── metrics.py      #   缺失区 PSNR/SSIM
│   ├── interpolation.py#   最近邻插值基线
│   ├── models/         #   Matrix/CP/Tucker 统一模型 + registry
│   ├── trainer.py      #   固定 Trainer（train/val 选步数 + 全量重训 + ModelBuilder）
│   ├── fair_experiment.py  # Day6 成对配置、GT 隔离调参、最终评估
│   └── experiment_judge.py # Day6 确定性晋升门禁 + 结构化反馈
├── agent_tools/        # Day3 把内核封装成 Hello Agents 工具
├── candidate/          # Day5/6 候选：schema、generator、validator、loader、晋升 registry
├── knowledge/          # Day4 本地 Matrix/CP/Tucker 知识库 + 检索器
├── method_selector.py  # Day4 方法选择（LLM + 规则 fallback）
├── workflow*.py        # Day3/4/5/6/full 各层工作流
├── run*.py             # 各 Day 的命令行入口
├── reporting.py        # Day7 标准化结果 + 生成 report.md
├── benchmark.py        # Day7 case 网格 + 按角色聚合
└── algorithms/         # 候选（candidates/）与晋升算法（approved/，版本化）
```

---

## 四、四个核心设计原则（面试重点）

**1. LLM 与确定性工具分工**
LLM 负责选择、解释、提 hypothesis、生成受约束代码；程序负责数据隔离、训练、指标、Judge、状态持久化。自然语言描述目标，程序化控制权限边界。

**2. 候选代码的可信执行链**
`schema 约束 → AST 策略检查 → 独立进程 smoke test → 代码哈希门控 → 固定 Trainer`。每道门解决不同风险（字段合法 / 无危险 import/call / 能训练 / 未被篡改 / 不越权）。

**3. 公平、无泄漏的实验协议**
基线与候选成对配置、预算一致；隐藏区域 GT 只在最终评估出现一次；候选是否成功只能由确定性 Judge 宣布（LLM 不当裁判）。

**4. 正确处理失败与证据边界**
候选失败时保存反馈并有限迭代（≤2 轮）；晋升后仍与强插值基线比较；单图结果不包装成 SOTA。

---

## 五、面试高频问题速答

1. **为什么不用缺失区域 GT 调参？** 因为 agent 会「偷看标准答案」过拟合评测样本。正确做法：调参只在观测像素内部的 train/val split 上进行，GT 只在最后评估出现一次。

2. **为什么不让 LLM 生成整个训练脚本？** 自由生成会重复已验证逻辑、易踩 shape/初始化错误，且无法执行权限边界。改为：候选继承已选基类、只覆盖允许改动，配合 AST + smoke + 哈希门控。

3. **为什么候选晋升后最终输出仍可能是插值？** 晋升只证明候选「比张量基线好」，夺冠还要「比插值基线好」。晋升门禁和冠军选择是两个独立机制。

4. **AST 检查为什么不是安全沙箱？** AST 只保证语法与黑白名单，可被复杂表达式绕过；独立进程提供崩溃隔离和超时，但仍运行在用户权限下。生产需容器/微虚拟机/只读文件系统等。

5. **为什么改进循环必须有最大轮数？** 保证停机、限制算力/时间成本、避免对单图过拟合；当前设为 1–2 轮。

---

## 六、关键术语

| 术语 | 含义 |
|---|---|
| GT（Ground Truth） | 完整图，藏于 `evaluation_ground_truth.png`，只在最终评估使用 |
| mask | `True`=观测可见，`False`=缺失待修 |
| corrupted | 缺损图（mask 应用到 GT 得到） |
| ImageProfile | 只由可见像素算出的图像画像（不泄漏 GT） |
| MethodPlan | Pydantic 校验的方法选择方案 |
| CandidateProposal | 候选改进方案（含假设、风险、搜索空间、完整代码） |
| ModelBuilder | `(shape, mean, hp) → nn.Module` 的构造协议，让动态候选旁路内置 registry |
| Judge | 确定性晋升门禁（PSNR 提升达标 + SSIM 不劣化 + trial 预算相等） |
| promotion | 版本化晋升到 `algorithms/approved/`（v1/v2/…，不可变可审计） |
