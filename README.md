# omniVLM

轻量多模态大模型训练与对齐：从 65M 全流程（Pretrain → SFT → GRPO → 评测）到 Qwen2.5-VL-3B LoRA 规模实验，单卡 24GB 全流程可复现。

## 结论（2026-08-12）

| 基准 | 65M（最优） | Qwen2.5-VL-3B + LoRA |
|---|---:|---:|
| VQAv2（2000 题，同口径） | 20.9% | **82.0%** |
| COCO BLEU-4（val2017 全量） | 0.0241 | **0.2566** |
| COCO ROUGE-L | 0.2138 | **0.4542** |
| COCO CIDEr | 0.0067 | **0.8364** |
| MMBench en/dev | 12.0%* | **84.9%**（零样本 83.95%） |
| 训练成本 | ~10 分钟 | 28 分钟 / 0.98% 可训练参数 / 17.9GB |

\* 65M 为 200 题抽样；3B 为 4329 题全量。

GRPO 对齐修复：官方 BLEU-4 从 0.0067 恢复到 **0.0266（超过 SFT）**，KL 由末步 1.23 压至 0.045。

## 样例图

![caption-contrast](benchmark_results/official_coco_20260812/sample_caption_contrast.png)

Caption 视觉依赖对照：原图 / 正确图预测 / 全黑图预测 / 错配图预测（SFT 模型）。
