# Day 4 学习笔记：让 Agent 自主选择张量分解方法

## 今天真正要学会什么

Day 3 已经有工具、状态机和评估边界，但使用的模型始终是写死的 Tucker。Day 4 第一次让一个决策节点真正参与研究流程：

```text
Day 3: selected_model = "tucker"
Day 4: ImageProfile + Knowledge Retrieval + LLM/Rules → MethodPlan
```

今天需要掌握：

1. 如何把图片转换成不会泄漏隐藏 Ground Truth 的结构化画像。
2. 如何建立可检查、可修改的本地方法知识库。
3. 如何进行不依赖向量数据库的简单检索。
4. 如何用 Pydantic 限制 LLM 的研究决策。
5. 如何处理 LLM JSON 错误、证据幻觉和 API 不可用。
6. 如何让选择结果自动驱动后续调参与训练。

## 1. Day 4 的决策链

```text
corrupted image + mask
          │
          ▼
     ImageProfile
          │
          ├──────────────┐
          ▼              ▼
 selection_rules.yaml   Markdown 方法文档
          │              │
          └──── LocalKnowledgeRetriever
                         │
                         ▼
               带来源的 evidence chunks
                         │
                         ▼
                  MethodSelector
                    │         │
               LLM 可用     LLM 不可用/连续非法
                    │         │
                    ▼         ▼
                JSON 校验   确定性规则回退
                    └────┬────┘
                         ▼
                    MethodPlan
                         │
                         ▼
                 Tune → Train → Evaluate
```

选择器从不接收完整图、缺失区域 PSNR 或最终 SSIM。最终指标只有模型训练结束后才出现。

## 2. 扩展后的 ImageProfile

`AnalyzeImageTool` 现在提供：

| 特征 | 含义 | 可能影响的选择 |
|---|---|---|
| `image_shape` | 高、宽、通道数 | 限制合法 rank |
| `actual_missing_rate` | 实际缺失比例 | 判断任务难度 |
| `mask_type` | random 或 block | 随机缺失与外推任务不同 |
| `missing_component_count` | 四连通缺失区域数 | 区分一个大洞与许多小洞 |
| `largest_missing_component_image_ratio` | 最大洞占全图比例 | 判断连续外推难度 |
| `image_aspect_ratio` | 长边/短边 | 判断空间各向异性 |
| `visible_mean_absolute_channel_correlation` | 可见 RGB 相关性 | 高相关时 Tucker 小通道秩有意义 |
| `visible_local_smoothness_score` | 可见邻域平滑程度 | 平滑图可能适合更简单低秩模型 |
| `visible_high_frequency_energy_ratio` | 可见邻接差分能量比例 | 高频纹理需要更高空间容量 |

### 为什么强调 visible

通道相关性、平滑度和高频能量只在以下位置计算：

```text
observed_mask == True
```

局部差分也只使用两个端点都可见的像素对。人工隐藏区域不会因为“只是做统计”而被提前读取。

### 高频指标不是完整频谱

当前 `visible_high_frequency_energy_ratio` 使用可见相邻灰度差的平方能量，并除以可见灰度能量。它是轻量代理指标，不是完整图像的 Fourier 高频占比。这样做是因为对不规则观测区域直接计算 FFT 会混入 mask 边界伪影。

## 3. 本地方法知识库

知识库位于：

```text
knowledge/
├── selection_rules.yaml
└── methods/
    ├── matrix_factorization.md
    ├── cp_decomposition.md
    └── tucker_decomposition.md
```

每份方法文档都包含：

- 数学形式；
- 参数量特点；
- 优势和限制；
- 缺失模式经验；
- 空间结构和通道相关性经验；
- 推荐 rank；
- 常见失败现象。

LLM 不是只凭预训练记忆做选择，而是必须引用这些项目内证据。

### 结构化规则

规则示例：

```yaml
- condition: channel_correlation_high
  prefer: tucker
  weight: 2.5
  reason: Tucker can allocate a small channel rank while retaining separate spatial ranks.
```

规则有两个作用：

1. 帮助检索器找到更相关的文档片段。
2. LLM 不可用时给出可复现的回退选择。

规则是经验先验，不是真理。最终模型是否优秀仍然只能由统一实验决定。

## 4. 简单本地检索如何工作

`LocalKnowledgeRetriever` 不使用 embedding 或向量数据库，流程是：

1. 根据 ImageProfile 激活 YAML 规则。
2. 按 Markdown 标题切分方法文档。
3. 从固定查询、规则关键词和偏好方法生成关键词集合。
4. 对每个片段计算关键词匹配分数。
5. 给被激活规则偏好的方法增加分数。
6. 返回 Top-K 片段及其来源。

一个 evidence source 类似：

```text
tucker_decomposition.md#Common failure modes
```

这种检索很简单，但有三个适合一周项目的优点：

- 不需要额外服务；
- 每个结果可以人工解释；
- 修改文档后行为容易调试。

Day 4 的目标是学习 Agent 结构，不是先搭建复杂 RAG 基础设施。

## 5. MethodPlan Schema

LLM 必须输出满足以下结构的 JSON：

```text
method                      matrix / cp / tucker
reason                      与 profile 或 evidence 对应的理由
evidence[]                  source + claim
confidence                  [0, 1]
suggested_hyperparameters   有界候选 rank 和初始化建议
risks[]                     当前选择的已知风险
selection_mode              llm / llm_repaired / deterministic_fallback
```

Pydantic 额外执行：

- 禁止未声明字段；
- 方法只能三选一；
- confidence 必须在 `[0,1]`；
- reason、evidence、risks 不能为空；
- 超参数建议不能为空。

代码还会检查每个 `evidence.source` 是否真的出现在本次检索结果或激活规则中。LLM 即使生成格式正确但来源不存在的 JSON，也会被拒绝。

## 6. 一次修复和确定性回退

选择器有三条执行路径。

### 路径 A：首次输出合法

```text
LLM → valid MethodPlan → selection_mode=llm
```

### 路径 B：首次非法，修复成功

```text
LLM → invalid JSON
    → 把具体校验错误反馈给同一个 LLM
    → valid MethodPlan
    → selection_mode=llm_repaired
```

只修复一次，避免 Agent 在格式错误上无限循环和浪费 token。

### 路径 C：API 不可用或连续非法

```text
activated rules
→ 按 method 累加 weight
→ 选择最高分方法
→ 生成带规则证据的 MethodPlan
→ selection_mode=deterministic_fallback
```

因此 API Key 缺失不会阻止本地开发、测试和演示。

## 7. LLM 建议不会直接成为任意训练参数

即使 MethodPlan 通过 Schema，超参数仍然属于不可信输入。Day 4 工作流会再次约束：

- rank 必须是正整数；
- Matrix rank 不超过矩阵合法上限；
- Tucker rank 不超过对应图像维度；
- RGB channel rank 不超过 3；
- 学习率限制在 `[1e-5, 1]`；
- 初始化尺度限制在 `(0,1]`；
- 候选数量最多 3 个。

然后 `TuneTensorModelTool` 仍然只用 held-out observed MSE 选择配置。LLM 只能提出一个小型搜索空间，不能宣布哪个配置最终最好。

## 8. 如何配置 LLM

项目沿用 Hello Agents 的环境变量：

```bash
export LLM_MODEL_ID="your-model-name"
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

对于本地 vLLM 的 OpenAI-compatible server，`LLM_BASE_URL` 可以指向服务器的 `/v1` 地址；具体模型名必须与服务器实际提供的模型一致。

三种运行模式：

```text
--llm-mode auto      配置完整则调用 LLM，否则规则回退
--llm-mode off       强制规则回退，适合测试
--llm-mode required  缺少配置或调用失败时明确报错/进入修复回退逻辑
```

当前 `required` 会要求环境变量完整；模型返回内容仍然要经过 Schema 校验。

## 9. 运行 Day 4

不调用 LLM，先验证本地闭环：

```bash
python3 -m inpainting_research_agent.run_day4 \
  --image inpainting_research_agent/assets/example.png \
  --mask-type block \
  --missing-rate 0.4 \
  --seed 42 \
  --image-size 128 \
  --max-steps 200 \
  --device auto \
  --llm-mode off
```

配置 LLM 后：

```bash
python3 -m inpainting_research_agent.run_day4 \
  --image inpainting_research_agent/assets/example.png \
  --device auto \
  --llm-mode required
```

## 10. 当前真实运行结果

本次运行 ID：`day4-20260901-160623-fcda33`。

画像关键值：

```text
mask type:                 block
missing rate:              0.400024
missing components:        1
largest hole ratio:        0.400024
image aspect ratio:        2.0
RGB mean abs correlation:  0.579
local smoothness score:    0.811
```

激活规则：

| 规则 | 偏好 | 权重 |
|---|---|---:|
| block missing | Tucker | 2.0 |
| large contiguous hole | Tucker | 2.0 |
| high aspect ratio | Tucker | 1.5 |
| high local smoothness | Matrix | 1.0 |

因此确定性选择为：

```text
method: Tucker
confidence: 5.5 / 6.5 = 0.8462
selection_mode: deterministic_fallback
```

注意：RGB 相关性 0.579 没有达到 0.75，所以没有激活 `channel_correlation_high`。这说明理由来自实际特征，而不是事后编造。

MethodPlan 生成两个 Tucker trial：

| Trial | Rank | Validation MSE | best step |
|---|---|---:|---:|
| 0 | (8,12,2) | 0.00872537 | 120 |
| 1 | (12,16,3) | 0.00601591 | 150 |

最终选择 trial 1，并在全部观测像素上重新拟合 150 步。

| 方法 | Missing PSNR | Composite SSIM |
|---|---:|---:|
| 最近邻插值 | 15.0618 dB | 0.727073 |
| Tucker | 12.1241 dB | 0.651443 |

最终仍然是 `winner=baseline`，Tucker 没有晋升。选择器做出了有证据支持的基础方法选择，实验则诚实地证明它在当前配置下不如插值，两者并不矛盾。

## 11. 输出目录新增内容

Day 4 在 Day 3 产物基础上新增：

```text
retrieval_result.json   激活规则和带来源的 Top-K 文档片段
method_plan.json        最终结构化选择、修复记录和回退原因
```

`method_plan.json` 还明确记录：

```json
{
  "ground_truth_provided_to_selector": false,
  "final_metrics_provided_to_selector": false
}
```

Trace 新增 `method_selection` 事件，记录 method、reason、evidence、confidence 和 selection mode。

## 12. 为什么暂时不接 VLM

视觉分析是可选项。当前版本没有接入 vLLM/DepictQA 图像描述，原因是：

1. ImageProfile 已足以验证检索与选择闭环。
2. VLM 服务配置不应该阻塞 Agent 主流程。
3. 先验证结构化决策，再增加视觉语义信息，更容易定位效果来源。

后续可以新增 `VisualDescriptionTool`，只给它 corrupted image、mask overlay 和 interpolation result，让它描述纹理、边缘、重复结构和语义区域，但不能预测 PSNR/SSIM，也不能看完整图。

## 13. 测试覆盖

当前共有 27 项测试，新增覆盖：

- 检索返回激活规则和带来源片段；
- 同一画像的确定性选择完全一致；
- 合法 LLM JSON 首次通过；
- 非法 JSON 只修复一次；
- 连续非法后回退到合法方法；
- Day 4 选择结果能自动训练对应模型；
- Selector 未收到 Ground Truth 或最终指标；
- Trace 包含 method selection 且没有评估专用路径。

## 14. 今日练习

### 练习 A：手算规则分数

打开 `image_profile.json` 和 `selection_rules.yaml`，不运行代码，手工判断哪些规则会激活并计算三个方法的总分，再与 `method_plan.json` 比较。

### 练习 B：改成 random mask

只把 `--mask-type` 改为 `random`，观察：

- 连通缺失区域数量如何变化；
- 激活规则如何变化；
- 最终选择是否变化。

### 练习 C：制造 LLM 格式错误

阅读 `FakeLLM` 测试，先返回普通文本，再返回合法 JSON，理解为什么第二次被标记为 `llm_repaired`。

### 练习 D：审计证据

逐条检查 `method_plan.json` 中的 evidence source 是否存在于 `retrieval_result.json`，并判断 claim 是否真的受到原文支持。

### 练习 E：质疑当前规则

当前规则因为一个 40% block hole 偏好 Tucker，但最终插值更好。提出一个不使用隐藏指标的新规则或验证策略，并说明如何公平验证它。

## 15. 今日完成标准

- 能解释 ImageProfile 每个新字段如何计算，以及为什么没有泄漏。
- 能说明本地检索为什么返回 source，而不只返回文本。
- 能写出 MethodPlan 的主要字段和约束。
- 能解释一次修复和确定性回退的执行条件。
- 能说明为什么 LLM 的 rank 建议仍需二次验证。
- 能从 Trace 找到方法选择理由和证据。
- 能解释“合理选择但实验失败”为什么是正常研究结果。

## 16. Day 5 会做什么

Day 4 只在已有 Matrix、CP、Tucker 中选择。Day 5 将把选中基础模型的代码、ImageProfile、方法证据和实验反馈交给 Model Improver，让 LLM 提出一个受限制的张量分解改进 idea，并生成可静态检查、可隔离执行的候选代码。
