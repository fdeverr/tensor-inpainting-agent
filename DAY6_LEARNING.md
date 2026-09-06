# Day 6：公平实验、失败反馈与算法晋升

## 1. 今天完成什么

Day 5 只回答了“候选代码能不能安全地进入训练”。Day 6 要回答更重要的问题：

> 在相同数据、相同训练预算和相同选择规则下，候选模型是否真的比基础张量分解更好？

今天完成了一条可以审计的研究闭环：

```text
读取已验证候选
→ 为基础模型和候选模型生成配对超参数
→ 只用 M_train 训练
→ 只用 M_val 选择超参数和训练步数
→ 在全部观测点 M 上重新拟合
→ 最后才读取缺失区域 Ground Truth
→ 固定 Judge 决定接受或拒绝
→ 失败时生成结构化反馈，最多再改进一轮
→ 成功时保存为带版本的 approved algorithm
→ 用同一个通用 Tool 加载和运行所有 approved algorithms
```

涉及的主要文件：

| 文件 | 职责 |
|---|---|
| `core/fair_experiment.py` | 配对试验、无泄漏调参、最终拟合与评估 |
| `core/experiment_judge.py` | 固定接受条件和结构化反馈 |
| `core/trainer.py` | 新增 `model_builder`，让固定 Trainer 可以运行动态候选类 |
| `candidate/loader.py` | 检查验证状态和代码哈希后加载候选 |
| `candidate/approved_registry.py` | 将通过 Judge 的候选保存为版本化算法 |
| `agent_tools/algorithm_runner.py` | 通用的已晋升算法运行 Tool |
| `workflow_day6.py` | 最多两轮的完整研究状态机 |
| `run_day6.py` | Linux 命令行入口 |

## 2. 为什么不能直接比较两次随便训练的结果

假设基础 Tucker 搜索 2 组配置、训练 200 步，而候选 Tucker+TV 搜索 10 组配置、训练 1000 步。即使候选指标更高，也无法判断提升来自模型 idea，还是来自更多计算资源。

Day 6 的公平性合同是：

- 两者的 trial 数相同，最多为 5；
- 最大训练步数、学习率、早停规则相同；
- 使用同一张图、同一个 mask、同一个随机种子和同一个设备；
- 每对 trial 的基础分解参数相同；
- 候选只额外拥有 idea 引入的参数，例如 `tv_weight`；
- 都依据同一份观测验证集的 MSE 选优。

例如第 4 对配置可能是：

```python
baseline = {
    "rank_h": 32,
    "rank_w": 32,
    "rank_c": 2,
    "init_scale": 0.2,
}

candidate = {
    "rank_h": 32,
    "rank_w": 32,
    "rank_c": 2,
    "init_scale": 0.2,
    "tv_weight": 0.0005,
}
```

因此，这一对实验之间唯一的算法差异就是 TV 正则。

## 3. 配对超参数是怎样产生的

`paired_trial_configurations()` 接收基础模型和候选模型各自声明的 `search_space`。

它先从基础搜索空间选择相同数量的配置，然后找出候选独有的参数：

```python
extra_space = {
    key: values
    for key, values in candidate_search_space.items()
    if key not in base_search_space
}
```

之后将候选独有参数逐个配到基础配置上。返回值同时记录：

- `baseline` 配置列表；
- `candidate` 配置列表；
- `shared_parameter_names`；
- `candidate_only_parameter_names`。

这些内容会写入每轮的 `paired_configurations.json`，所以以后可以检查 Agent 有没有偷偷给候选更多机会。

## 4. 如何从函数接口上隔离 Ground Truth

真正防止数据泄漏，不能只依赖 prompt 中的一句“不要看 Ground Truth”。更可靠的做法是让调参函数根本没有这个输入：

```python
def tune_model_on_observed_pixels(
    model_name,
    model_builder,
    configurations,
    observed_image,
    observed_mask,
    training_config,
    seed,
):
    ...
```

这里没有 `ground_truth` 参数。每个 trial 内部调用 Day 2 的固定 Trainer：

```text
M_observed → 固定随机划分 → M_train + M_val
M_train    → 梯度更新
M_val      → 选择 hyperparameters 和 best_step
```

调参结果明确保存：

```json
{
  "selection_metric": "held_out_observed_mse",
  "ground_truth_used": false
}
```

只有 `final_fit_and_evaluate()` 才接收 Ground Truth。此时配置和步数已经确定，模型也已经在全部观测点上重新训练完毕，因此隐藏像素的指标不能反向影响选择。

## 5. 为什么 Trainer 增加 `model_builder`

Day 2 的基础模型可以通过名称从固定字典中创建：

```python
create_model("tucker", ...)
```

但 Day 5 的候选类是运行时生成的，固定 Trainer 事先不知道其名称。如果为每个候选生成一份训练代码，Agent 就可能同时修改训练规则、评估规则甚至数据读取方式。

因此今天只给 Trainer 增加一个受限注入点：

```python
model_builder(image_shape, initial_channel_mean, hyperparameters)
```

候选只负责构造模型；数据划分、优化器、梯度检查、早停和评估仍完全由固定 Trainer 控制。这正是“允许 Agent 改模型，但不允许它改裁判”。

## 6. 候选为什么还要经过一次加载门

`load_validated_candidate()` 在动态执行代码前检查四件事：

1. `manifest.json`、`validation.json` 和 `model.py` 都存在；
2. manifest 状态为 `validated`；
3. `eligible_for_training` 和 validation 的 `passed` 都为真；
4. 当前 `model.py` 的 SHA-256 与验证时记录的哈希完全相同。

这可以阻止一种常见问题：候选通过检查后，代码又被修改，却继续使用旧的“验证通过”标签。

动态 `exec` 仍不是生产级安全沙箱。它在这个学习项目中建立的是完整性门和接口门，线上系统仍应使用容器、资源限制和更严格的进程隔离。

## 7. 固定 ExperimentJudge

最终结论不能由生成 idea 的 LLM 自己给出。`judge_candidate()` 使用硬编码准入条件：

```text
PSNR_candidate - PSNR_baseline >= 0.2 dB
SSIM_candidate - SSIM_baseline >= -0.002
baseline_trial_count == candidate_trial_count
```

Judge 输出的不只是 `accept/reject`，还包括：

```json
{
  "decision": "reject",
  "psnr_delta": -0.1,
  "ssim_delta": -0.003,
  "runtime_ratio": 1.2,
  "training_behavior": {},
  "suspected_causes": [],
  "next_round_constraints": []
}
```

其中 `runtime_ratio` 不参与本次硬门槛，但会被记录。这样面试时可以说明：当前 MVP 以质量为准入条件，同时保留了以后加入效率约束所需的数据。

## 8. 失败反馈怎样驱动第二轮

如果第一轮被拒绝，Workflow 会把以下内容放入 `previous_failure_feedback`：

- 决策和指标差值；
- 两者最佳验证误差和最佳步数；
- 可能的失败原因；
- 下一轮必须遵守的约束。

随后再次调用同一个 `CandidateGenerator`。真实 LLM 可以据此提出不同改动；当前未配置 LLM 时，确定性 fallback 会把 TV 搜索范围从：

```python
[0.0, 0.0001, 0.0005, 0.001]
```

缩小为：

```python
[0.0, 0.00001, 0.00005, 0.0001]
```

新代码仍须重新完成 AST 和独立进程 smoke test，不能因为它来自第二轮就跳过验证。

停止条件是确定的：

```text
候选通过 → 立即停止
候选未通过且还有轮次 → 生成下一候选
达到 max_improvement_rounds → 停止并返回最佳基线
```

配置强制 `max_improvement_rounds <= 2`，因此不会出现无限自我修改。

## 9. 算法晋升不是复制一个名字

通过 Judge 后，`promote_candidate()` 创建：

```text
algorithms/approved/<algorithm_name>/<version>/
├── model.py
├── idea.json
├── manifest.json
├── validation.json
├── best_config.json
├── comparison.json
└── approved_manifest.json
```

`approved_manifest.json` 记录来源候选、来源实验、代码哈希、适用条件、最佳配置和比较结论。版本号只递增，不覆盖过去实验。

这里要区分两个结论：

- **晋升结论**：候选是否显著优于它要改进的张量基础模型；
- **最终输出冠军**：插值基线、张量基线和已晋升候选中，谁的缺失区域 PSNR 最高。

一个候选可以正确证明 TV 改进了 Tucker，因此被晋升；但在当前图片上，简单插值仍可能是全体算法冠军。Research Agent 必须如实报告两件事，不能为了强调“创新”隐藏强基线。

## 10. 为什么只需要一个 AlgorithmRunnerTool

每晋升一个算法就自动生成一个新的 Tool 类，会导致工具数量不断膨胀，也让工具接口难以维护。

这里采用稳定接口：

```text
AlgorithmRunnerTool
  + approved_dir
  + corrupted_path
  + mask_path
  + ground_truth_path（只用于最终评估）
  + output_dir
```

Tool 从 manifest 读取算法名称、版本和最佳配置，重新检查代码哈希，再调用固定 Trainer。以后晋升 CP+Smoothness、Matrix+Regularization，也复用同一个 Tool。

这就是“新算法成为工具能力”，但不是“为每个算法复制一套训练代码”。

## 11. 本次真实运行结果

运行使用你的 `assets/example.png`、40% 连续块缺失、CPU、4 个配对 trials、每个最多 200 步。

| 方法 | 缺失区域 PSNR | Composite SSIM | 结论 |
|---|---:|---:|---|
| 最近邻插值 | 15.0618 dB | 0.7271 | 全体算法冠军 |
| 同预算 Tucker | 12.7903 dB | 0.6269 | 候选的比较基线 |
| Tucker + TV | 13.5133 dB | 0.7087 | 相对 Tucker 通过晋升 |

候选相对 Tucker：

```text
PSNR delta = +0.7230 dB
SSIM delta = +0.0818
runtime ratio = 1.34
selected tv_weight = 0.0005
```

因此第一轮就满足 `+0.2 dB / -0.002 SSIM` 门槛，第二轮没有执行。候选被保存为 `tucker_tv_regularized/v2`。之所以是 `v2`，是因为开发验证期间已经保存过一次同一候选的 `v1`，也验证了版本不会被覆盖。

必须谨慎解释结果：这只是当前一张图片和一种 mask 上的证据，不能声称该方法普遍优于 Tucker，更不能声称达到 SOTA。第 7 天应加入多案例报告或至少把这一限制清楚展示出来。

## 12. 运行命令

在 Hello Agents 仓库根目录执行：

```bash
python -m inpainting_research_agent.run_day6 \
  --base-run-dir inpainting_research_agent/outputs/<day4-run-id> \
  --candidate-dir inpainting_research_agent/algorithms/candidates/<candidate-id> \
  --llm-mode off \
  --tuning-trials 4 \
  --max-steps 200 \
  --max-improvement-rounds 2 \
  --device cuda
```

如果服务器暂时没有 CUDA，把最后一项改为 `cpu`。正式运行前先安装 `requirements.txt`。

## 13. 建议按这个顺序阅读代码

1. 先读 `paired_trial_configurations()`，手算一组 baseline/candidate 配对。
2. 再读 `tune_model_on_observed_pixels()`，确认函数签名中不存在 Ground Truth。
3. 对照 Day 2 阅读 Trainer 新增的 `model_builder`。
4. 阅读 `judge_candidate()`，手动代入本次三个指标。
5. 阅读 `Day6Workflow.run()`，画出接受与拒绝两个分支。
6. 最后阅读 `promote_candidate()` 和 `AlgorithmRunnerTool`，理解算法仓库与执行工具的分离。

## 14. 今天的实践练习

### 练习 1：复核公平配对

打开本次运行的 `round_1/paired_configurations.json`，逐对确认除 `tv_weight` 外的参数都完全相同。

### 练习 2：强制走失败分支

把命令中的门槛临时提高：

```bash
--minimum-psnr-delta 5.0
```

观察第一轮如何产生反馈、第二轮如何生成更小 TV 权重的新候选，以及两轮后如何停止。这个练习不要改默认门槛。

### 练习 3：解释“晋升但不是冠军”

用自己的话回答：为什么 Tucker+TV 可以被晋升，但最终补全结果仍输出最近邻插值？如果能明确区分“控制变量实验”和“生产输出选择”，就理解了今天最重要的设计。

### 练习 4：检查算法完整性

复制一个 approved 目录用于实验，修改其中 `model.py` 的任意字符，再调用 `AlgorithmRunnerTool`。它应因 SHA-256 不匹配而拒绝加载。

## 15. Day 6 完成标准

- [x] 基础模型与候选使用相同 trial 数和训练预算。
- [x] 调参函数不能接收隐藏 Ground Truth。
- [x] 候选只能通过受限 `model_builder` 接入固定 Trainer。
- [x] Judge 使用固定 PSNR/SSIM 门槛。
- [x] 失败反馈能进入下一轮生成输入。
- [x] 改进循环最多执行两轮。
- [x] 通过候选被保存到版本化 approved registry。
- [x] 一个通用 Tool 可以重新加载并运行 approved algorithm。
- [x] 最终报告同时展示插值、基础张量和候选结果。
- [x] 自动化测试全部通过。

完成 Day 6 后，项目已经具备 Research Agent 的核心闭环。Day 7 将把前六天串成一个统一入口，并整理求职展示所需的 README、实验报告和失败边界。
