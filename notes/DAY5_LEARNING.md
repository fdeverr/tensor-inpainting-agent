# Day 5 学习笔记：让 Agent 生成并验证候选算法

## 今天真正要学会什么

Day 4 的 Agent 只能从 Matrix、CP 和 Tucker 中选择已有方法。Day 5 开始让 Agent 提出研究改进并生成代码：

```text
已完成的 Day 4 基础实验
        │
        ▼
Model Improver
        │
        ├── hypothesis
        ├── proposed changes
        ├── expected effect
        ├── risks
        ├── search space
        └── complete model code
        │
        ▼
CandidateProposal Schema
        │
        ▼
AST static validation
        │ passed
        ▼
independent-process smoke test
        │ passed
        ▼
eligible_for_training = true
```

今天最重要的不是“让 LLM 写出复杂模型”，而是学习如何把代码生成变成一个有假设、有边界、有验证、有失败反馈的工程环节。

## 1. 为什么 Day 5 从已完成的 Day 4 run 开始

运行入口接收：

```text
--base-run-dir outputs/<completed-day4-run>
```

Model Improver 会读取：

- `ImageProfile`；
- MethodPlan 和方法选择理由；
- 基础模型源代码；
- 基础模型最佳配置；
- 最终拟合曲线摘要；
- 基础模型 PSNR/SSIM；
- 插值基线 PSNR/SSIM；
- 比较结果；
- 之前的失败反馈。

不会提供：

- 完整 Ground Truth 图片内容；
- 评估函数源代码的修改权限；
- shell、网络或外部程序权限。

### 关于基础指标的边界

Day 5 允许 Improver 看到基础算法已经产生的聚合指标，因为它需要知道基础方法失败了多少以及形成改进假设。但如果反复针对同一张图片、同一个人工洞根据最终 PSNR 改代码，就会逐渐对评测样本过拟合。

正式研究应进一步划分：

```text
development images/masks   用于 Agent 形成和改进 idea
final benchmark images     只在算法冻结后评估一次
```

一周 MVP 当前使用单图演示，因此必须在报告中承认这个限制。

## 2. 允许与禁止的改进

允许的改进仍然位于张量分解框架内：

- 因子初始化；
- 非对称 rank；
- 因子正则化；
- Total Variation；
- 因子平滑；
- rank 或正则权重调度；
- 保持张量参数化的多尺度拟合。

禁止：

- 使用预训练扩散模型；
- 用 CNN 或 Transformer 替代张量分解主体；
- 读取完整 Ground Truth；
- 修改评估代码；
- 调用网络、shell、子进程或外部程序；
- 通过增加训练预算获得不公平优势。

这些限制同时出现在 Prompt 和验证器中。只写在 Prompt 里不够，因为 LLM 可能误解或忽略自然语言要求。

## 3. CandidateProposal Schema

候选必须通过 Pydantic `CandidateProposal`：

```text
base_method          matrix / cp / tucker
hypothesis           可证伪的改进假设
proposed_changes     具体修改列表
expected_effect      预期影响
risks                可能失败的原因
search_space         有限超参数候选
model_code           完整 Python 模型代码
generation_mode      llm / llm_repaired / deterministic_template
```

Schema 禁止额外字段，并限制文本、列表、代码长度和空搜索空间。LLM 首次输出非法时允许修复一次，连续非法或没有配置 LLM 时使用确定性模板。

对应代码：`candidate/schemas.py` 和 `candidate/generator.py`。

## 4. 当前确定性候选的研究假设

Day 4 的基础 Tucker 在连续缺失块内容易出现条带或不连续变化。因此默认候选提出：

```text
Hypothesis:
在保留 Tucker 核心和因子参数的前提下，加入较小的图像空间 TV 正则，
可能抑制洞内条带和突变，提高空间连贯性。
```

损失变成：

```text
L = L_observed
  + λ_tv (mean|X[i+1,j]-X[i,j]| + mean|X[i,j+1]-X[i,j]|)
```

候选类继承基础 Tucker：

```python
class CandidateTensorInpaintingModel(TuckerDecomposition):
    ...
```

这意味着核心张量、空间因子、通道因子仍然全部是基础模型中的 `nn.Parameter`。TV 只改变优化目标，没有把张量分解替换成其他网络。

搜索空间：

```text
tv_weight ∈ {0, 1e-4, 5e-4, 1e-3}
```

其中 `tv_weight=0` 很重要：它把原始基础模型包含在候选搜索空间中，便于判断 TV 是否真的有帮助。

## 5. 为什么让候选继承基础模型

完全自由生成整套张量分解代码会重复很多已经验证的逻辑，也更容易产生 shape、初始化和参数注册错误。

当前策略是：

1. 验证进程预先提供 `MatrixFactorization`、`CPDecomposition`、`TuckerDecomposition` 和 `BaseTensorInpaintingModel`。
2. 候选代码不需要导入项目内部路径。
3. 候选可以继承已选基础模型，只覆盖允许修改的行为。
4. smoke test 最终检查它确实是 `BaseTensorInpaintingModel` 的子类。

这是计划中“如果自由代码生成不稳定，就提供基础模板”的具体实现。

## 6. 第一级验证：Python AST 静态检查

验证器先调用：

```python
ast.parse(candidate_source)
```

### Import 白名单

只允许：

```text
torch
torch.nn
torch.nn.functional
math
```

拒绝包括：

```text
os
subprocess
socket
requests
shutil
pathlib
urllib
http
importlib
```

### 危险调用黑名单

拒绝：

```text
open()
eval()
exec()
compile()
__import__()
input()
```

### 接口检查

AST 中必须存在：

- `CandidateTensorInpaintingModel`；
- 合法基础类；
- 候选类自己声明的 `@classmethod search_space()`。

静态检查失败时不会进入动态加载，`smoke_test.skipped=true`。

## 7. 第二级验证：独立进程 smoke test

通过 AST 后，控制器使用参数列表启动独立 Python：

```text
python -I candidate/smoke_runner.py algorithms/candidates/<id>/model.py
```

关键约束：

- `shell=False`；
- stdin 关闭；
- stdout/stderr 捕获；
- API key、token、secret 环境变量移除；
- CUDA 对 smoke test 隐藏；
- 设置短超时；
- 使用 `-I` 隔离用户 Python 环境。

子进程在 `16×16×3` 假数据上验证：

1. 动态加载候选类；
2. 检查继承关系；
3. 执行 forward；
4. 检查输出 shape、浮点 dtype 和有限值；
5. 计算 masked loss；
6. 执行 backward；
7. 检查至少一个参数获得有限梯度；
8. 执行一次 Adam update；
9. 检查至少一个参数确实发生改变；
10. 调用 `search_space()` 并检查非空字典。

## 8. 为什么独立进程仍然不是安全沙箱

独立进程提供崩溃隔离和超时，但它仍然运行在当前操作系统用户权限下。AST 黑白名单也可能被复杂 Python 表达式绕过。

因此当前方案只是 MVP guardrail，不适合直接运行互联网上的恶意代码。生产环境至少还需要：

- 容器或微型虚拟机；
- 只读文件系统；
- 独立低权限用户；
- 无网络 namespace；
- CPU、内存、进程数和磁盘配额；
- 更成熟的代码策略或仅允许 DSL/受限模板。

Day 5 的目标是理解验证层级，不是假装 AST 等于安全执行环境。

## 9. Candidate manifest 是训练门禁

每个候选目录：

```text
algorithms/candidates/<candidate-id>/
├── idea.json
├── model.py
├── manifest.json
└── validation.json
```

`manifest.json` 记录：

- candidate ID；
- base run 和 base method；
- 生成时间与生成模式；
- LLM 模型和 Prompt 版本；
- SHA-256 代码哈希；
- 允许的搜索空间；
- validation status；
- `eligible_for_training`。

只有：

```text
static_validation.passed == true
AND smoke_test.passed == true
```

才会设置：

```json
{
  "validation_status": "validated",
  "eligible_for_training": true
}
```

Day 6 的训练入口必须检查这个 manifest，不能只因为 `model.py` 存在就执行候选。

## 10. 如何运行 Day 5

先完成一次 Day 4，再把 run 目录交给 Day 5：

```bash
python3 -m inpainting_research_agent.run_day5 \
  --base-run-dir inpainting_research_agent/outputs/<day4-run-id> \
  --llm-mode off \
  --smoke-timeout 10
```

配置 Hello Agents LLM 后：

```bash
python3 -m inpainting_research_agent.run_day5 \
  --base-run-dir inpainting_research_agent/outputs/<day4-run-id> \
  --llm-mode required
```

## 11. 当前真实候选

基础实验：

```text
day4-20260901-160623-fcda33
base method: Tucker
```

生成结果：

```text
workflow:  day5-20260903-151313-f893a2
candidate: candidate-20260903-151313-6d4d12
mode:      deterministic_template
idea:      Tucker + image-space TV regularization
```

验证结果：

```text
AST parse:                    passed
import and call policy:       passed
candidate contract:           passed
forward shape:                [16,16,3]
dtype:                        torch.float32
loss terms:                   data_loss, tv_regularization
finite-gradient parameters:   5
optimizer changed parameter:  true
validation status:            validated
eligible for training:        true
```

注意：这只说明候选代码“合法且能训练”，不说明它比基础 Tucker 更好。Day 5 没有对候选进行正式调参，也没有计算候选 PSNR/SSIM。

## 12. 结构化失败反馈

验证失败时，`validation.json` 会记录稳定错误类型。例如：

```text
FORBIDDEN_IMPORT       使用了 os/subprocess 等模块
FORBIDDEN_CALL         使用了 open/eval/exec 等调用
MISSING_CLASS          缺少规定候选类
INVALID_BASE_CLASS     没有继承张量模型接口
MISSING_SEARCH_SPACE   缺少搜索空间
TimeoutExpired         动态测试超时
ValueError             forward shape 错误
RuntimeError           没有参数梯度或优化器没有更新参数
```

这些 feedback 会在 Day 6 进入下一轮 Improver 上下文，形成“失败—修复—再验证”的闭环。

## 13. 测试覆盖

当前共有 31 项测试，Day 5 新增验证：

- Tucker + TV 候选通过两级检查；
- 包含 `import os` 的候选在执行前被拒绝；
- 输出二维 tensor 的候选被 smoke test 拒绝；
- 参数不参与 forward 的候选因没有梯度被拒绝；
- 成功候选的 manifest 被标记为 eligible；
- 失败结果包含结构化 feedback。

运行：

```bash
python3 -m pytest \
  --rootdir=inpainting_research_agent \
  inpainting_research_agent/tests \
  -q
```

## 14. 今日练习

### 练习 A：阅读生成上下文

查看 `load_improver_context()`，列出提供给 LLM 的信息和刻意排除的信息。思考为什么可以提供聚合指标，但不提供 Ground Truth 图片。

### 练习 B：审计候选代码

逐行阅读 `model.py`，回答：

1. 哪些参数仍然来自 Tucker？
2. TV 项会对观测区和缺失区分别产生什么梯度？
3. 为什么 `tv_weight=0` 应包含在搜索空间中？

### 练习 C：制造静态失败

复制候选并增加 `import requests`，运行 Validator，确认 smoke test 没有执行。

### 练习 D：制造动态失败

保留合法 import，但让 forward 返回 `[H,W]`，观察静态检查通过、动态检查失败。这说明为什么只做 AST 不够。

### 练习 E：设计另一个张量改进

从以下方向任选一个，只写 hypothesis 和 search space，暂时不写代码：

- Tucker 因子相邻行平滑；
- 对 CP 因子做 L2 归一化约束；
- 根据图像长宽比设置非对称 rank；
- TV 权重随训练 step 调度。

## 15. 今日完成标准

- 能解释 CandidateProposal 的每个字段。
- 能区分“Schema 合法”“静态安全检查通过”“模型可训练”“指标更好”四种不同结论。
- 能说明 AST 检查和独立进程分别解决什么问题。
- 能说明当前安全方案为什么不是生产级沙箱。
- 能从 manifest 判断候选能否进入 Day 6 训练。
- 能解释 TV 候选的假设、预期收益和过平滑风险。
- 能独立运行 Day 5 并找到 idea、model、manifest 和 validation。

## 16. Day 6 会做什么

Day 6 将加载 `eligible_for_training=true` 的候选，在与基础方法相同的 trial 数和训练预算下完成：

```text
candidate tuning
→ final refit on all observed pixels
→ PSNR/SSIM evaluation
→ fair comparison with baseline
→ promote or reject
→ structured feedback for another improvement round
```

只有到那时，才能判断今天生成的 TV 候选是否真的有效。
