# omniVLM

轻量多模态大模型训练与对齐：从 65M 全流程（Pretrain → SFT → GRPO → 评测）到 Qwen2.5-VL-3B LoRA 规模实验，单卡 24GB 全流程可复现。

## 摘要表（2026-08-12）

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

## 关键实验与发现

1. **GRPO 奖励错位诊断与修复**：代理奖励上升 23.5% 但指标退化、KL 扩大 11×；通过自适应 KL、group 4→16、CIDEr 对齐奖励，官方 BLEU-4 恢复到 0.0266 并超过 SFT。
2. **VQA 格式对齐闭环**：基线 0 → VQA 专项 SFT 31.6%（暴露 caption 灾难性遗忘）→ VQA+caption 混合 SFT 20.9% 且 caption CIDEr 0.0067 全场最优。
3. **数据配比消融**：VQA:caption = 1:3 / 1:1 / 3:1，VQA 8.4% / 20.9% / 23.0%，caption 保持高位——多任务数据配比权衡可量化。
4. **规模跃迁**：3B LoRA（0.98% 参数、28 分钟）VQAv2 82.0%、MMBench 84.9%、COCO CIDEr 0.8364；零样本 vs LoRA 消融显示基座能力主导综合基准。
5. **数据工程**：sft 数据画像（中英均衡、去重率 86.1%、答案长度分布），发现 ~14% 近似重复并提出 minhash 去重方案。

## 技术栈与管线

- 架构：SigLIP2 视觉编码器 + MLP Projection + MiniMind LLM（65M）；Qwen2.5-VL-3B + LoRA（r=16）
- 训练：Pretrain（1.27M）→ SFT（2.9M）→ GRPO（5000 步）；原子断点续训、JSON 配置校验、GitHub Actions CI
- 评测：官方 COCOEvalCap（val2017 全量）、VQAv2（2000 题）、MMBench en/dev（4329 题）；black/shuffled 视觉依赖对照
- 硬件：单卡 RTX 4090（24GB），bf16

## 目录

```text
trainer/          训练脚本（pretrain / sft / grpo）
scripts/          评测与数据管线（COCO / VQA / MMBench / 数据画像 / 混合数据）
evaluation/       指标与 GRPO 奖励实现
benchmark_results/官方评测结果与报告
docs/             实验思路、训练岗对照、简历条目
```

## 文档

- [完整实验报告](benchmark_results/official_coco_20260812/REPORT.md)
- [实验思路与预期目标](docs/EXPERIMENT_OVERVIEW.md)
- [多模态训练岗要求对照](docs/TRAINING_ROLE_REVIEW.md)
- [简历项目条目](docs/RESUME_ENTRIES.md)
