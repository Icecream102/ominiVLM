# MiniMind-V 完整训练实验报告（2026-08-11）

## 实验范围

在单张 NVIDIA RTX 5090 上严格执行以下流水线：

1. Pretrain：`pretrain_i2t.parquet`，2 epochs，batch size 128，仅训练 Projection。
2. SFT：`sft_i2t.parquet`，2 epochs，batch size 64，训练 Projection 与 LLM 首尾层。
3. GRPO：从 SFT 数据固定抽取 5000 条，group size 4，PPO epochs 2，`beta=0.02`。
4. COCO2017 validation 固定抽取 500 张图片，对三个检查点分别执行 correct、black、shuffled 三条件统一评测。

Prompt 为 `<image>\nDescribe this image in one concise sentence.`，贪心解码，最多生成 48 tokens。指标为仓库内自包含的精确 token 实现，适合检查点间受控比较，不等同于官方 COCOEvalCap 论文数字。

## 训练摘要

| 阶段 | 规模 | 最终训练信号 | 耗时 | 峰值显存 |
|---|---:|---:|---:|---:|
| Pretrain | 2 × 9959 steps | loss 2.4652 | 1.688 h | 18.862 GB |
| SFT | 2 × 45383 steps | loss 1.9379 | 6.282 h | 18.111 GB |
| GRPO | 5000 steps | reward 末 500 步均值 0.2682 | 0.498 h | 见原始日志 |

GRPO 全程 reward 均值为 0.2559，前 500 步均值 0.2172，末 500 步均值 0.2682；但 KL 均值为 0.5500，并由前 500 步的 0.0761 上升到末 500 步的 0.8380。最终单步 reward 为 0.1904、KL 为 1.2291。

## COCO500 正确图结果

| 检查点 | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR-exact | ROUGE-L | CIDEr-style | tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrain | **0.2786** | **0.1502** | **0.0736** | **0.0366** | 0.2978 | **0.2563** | **0.01390** | **206.5** |
| SFT | 0.2767 | 0.1450 | 0.0704 | 0.0346 | **0.2995** | 0.2555 | 0.01179 | 198.1 |
| GRPO | 0.1694 | 0.0721 | 0.0257 | 0.0104 | 0.2012 | 0.1638 | 0.00028 | 178.5 |

## 视觉依赖消融

| 检查点 | 条件 | BLEU-4 | ROUGE-L | 输出变化率 | Token Jaccard 距离 |
|---|---|---:|---:|---:|---:|
| Pretrain | black | 0.0000 | 0.1926 | 100% | 0.7832 |
| Pretrain | shuffled | 0.0065 | 0.2074 | 100% | 0.7807 |
| SFT | black | 0.0000 | 0.1647 | 100% | 0.8720 |
| SFT | shuffled | 0.0069 | 0.2086 | 100% | 0.8020 |
| GRPO | black | 0.0000 | 0.1560 | 100% | 0.5579 |
| GRPO | shuffled | 0.0000 | 0.1523 | 100% | 0.5622 |

## 结论

- 正式 SFT 仅在 METEOR-exact 上略高于 Pretrain，其余主要 Caption 指标小幅下降，因此不能宣称 SFT 全面提升。可能原因包括 290 万混合 SFT 样本与 COCO Caption 目标的域差异，以及两轮训练后生成风格偏离简洁英文描述。
- GRPO 的代理奖励从前 500 步到末 500 步提高约 23.5%，但 COCO Caption 指标显著下降，同时 KL 扩大约 11 倍。这表明当前词面重叠、长度与格式组合奖励和目标基准不充分对齐，且 `beta=0.02` 的参考模型约束不足。
- GRPO 结果是一个明确的负实验：训练目标得到了优化，但通用 Caption 能力退化。后续应采用独立 reward-model/多指标奖励、增大或自适应 KL 系数、加入 held-out early stopping，并先做 `beta`、学习率和训练步数消融。
- 三个模型在 black/shuffled 条件下输出均发生变化，说明输出使用了视觉输入；不过输出变化率本身不能代表正确视觉理解，仍需结合质量下降和更强的 VQA/幻觉评测。

## 结果文件

- 三个模型的 `summary.json` 与逐样本预测位于 `results/full_pipeline/coco500/`。
- Pretrain、SFT、GRPO 和评测日志位于 `logs/`。
- 四个阶段的 `.done` 标记均已回传，可用于审计流水线完成状态。
