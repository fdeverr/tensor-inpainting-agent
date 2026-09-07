# 2～3 分钟项目演示提纲

## 0:00–0:20：问题和目标

“这是一个张量分解图像补全 Research Agent。它不只是调用一个模型，而是分析数据、检索方法经验、提出可证伪的改进，并用公平实验决定是否接受候选。”

展示 README 顶部架构图，强调 LLM 负责提出假设，固定工具负责训练与裁判。

## 0:20–0:45：一条命令启动

```bash
python -m inpainting_research_agent.run \
  --image inpainting_research_agent/assets/example.png \
  --mask-type block \
  --missing-rate 0.4 \
  --max-improvement-rounds 2 \
  --llm-mode off \
  --device auto
```

说明 `off` 是可复现 fallback，服务器连接模型 API 时使用 `required`，CUDA 由 `auto` 自动选择。

## 0:45–1:20：方法选择

打开最终 `report.md`：

- 展示实际缺失率、缺失连通区域、通道相关性和局部平滑度；
- 展示从本地知识文档检索出的证据；
- 说明本例选择 Tucker 是因为连续块缺失和空间各向异性。

## 1:20–1:50：候选与公平实验

打开候选 `idea.json`、`validation.json` 和一轮 `paired_configurations.json`：

- idea 是 Tucker + TV；
- 代码先通过 AST 和独立进程训练 smoke test；
- baseline/candidate 各 4 个 trial，配对配置只差 `tv_weight`；
- 调参函数没有 Ground Truth 参数。

## 1:50–2:20：指标和图片

展示 corrupted、插值、Tucker、Tucker+TV 四张图和结果表：

```text
Tucker + TV 相对 Tucker：PSNR +0.7230 dB，SSIM +0.0818
```

解释候选因此被晋升，但插值的绝对 PSNR 仍最高，所以最终输出插值。这证明 Agent 能诚实地区分“假设成立”和“全体冠军”。

## 2:20–2:40：工程决策与边界

结束语：

“这个项目的价值是可审计的研究闭环，而不是在单图上声称 SOTA。下一步会加入多图片、多 mask、多随机种子的晋升协议，以及真正的容器级代码沙箱和 VLM 图像分析。”

## 录制前检查

- 终端中不显示 API Key；
- 使用一个干净输出目录；
- 提前确认 CUDA 和依赖；
- 浏览器预先打开 README、report、Trace HTML 和结果图片；
- 视频中明确说当前 benchmark 的图片数量；
- 控制在 3 分钟以内。
