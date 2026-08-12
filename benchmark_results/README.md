# COCO2017 MiniMind-V benchmark

## Official COCOEvalCap run (2026-08-12)

Full val2017 (5000 images) scored with official COCOEvalCap (pycocoevalcap).
Report and per-checkpoint summaries: [`official_coco_20260812`](official_coco_20260812/REPORT.md).

| Checkpoint | BLEU-4 | ROUGE-L | CIDEr |
|---|---:|---:|---:|
| Pretrain (full) | **0.0250** | **0.2160** | **0.0063** |
| SFT (full) | 0.0237 | 0.2136 | 0.0058 |
| GRPO (original) | 0.0067 | 0.1517 | 0.0002 |
| GRPO (fix: group 8 + adaptive KL) | 0.0201 | 0.2016 | 0.0016 |

The GRPO-fix variant recovers BLEU-4 by +200% vs the original GRPO with KL
held at ~0.07 instead of 1.23.

## Full Pretrain -> SFT -> GRPO run (2026-08-11)

The formal two-epoch Pretrain and SFT run, 5000-step GRPO run, three-checkpoint COCO500 evaluation, per-sample predictions, and raw logs are archived in [`full_pipeline_20260811`](full_pipeline_20260811/REPORT.md).

| Checkpoint | BLEU-1 | BLEU-4 | METEOR-exact | ROUGE-L | CIDEr-style |
|---|---:|---:|---:|---:|---:|
| Pretrain (full) | **0.2786** | **0.0366** | 0.2978 | **0.2563** | **0.01390** |
| SFT (full) | 0.2767 | 0.0346 | **0.2995** | 0.2555 | 0.01179 |
| GRPO (full) | 0.1694 | 0.0104 | 0.2012 | 0.1638 | 0.00028 |

The GRPO proxy reward improved during optimization, but held-out COCO caption quality degraded and KL increased substantially. The formal report records this as reward misalignment rather than a model-quality improvement.

Raw top-level summaries:

- [`coco500_full_pretrain_summary.json`](coco500_full_pretrain_summary.json)
- [`coco500_full_sft_summary.json`](coco500_full_sft_summary.json)
- [`coco500_full_grpo_summary.json`](coco500_full_grpo_summary.json)

## Earlier pilot benchmark

## Setup

- Hardware: one NVIDIA RTX 5090
- Dataset: 500 images sampled from COCO2017 validation with seed 42
- Checkpoints: native PyTorch `pretrain_vlm_768.pth` and `sft_vlm_768.pth`
- Prompt: `<image>\nDescribe this image in one concise sentence.`
- Decoding: greedy, at most 48 new tokens
- Controls: correct image, black image, deterministically shuffled image
- Model size: 65.09M parameters

## Correct-image results

| Checkpoint | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR-exact | ROUGE-L | CIDEr-style | tokens/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrain | 0.2827 | 0.1568 | 0.0810 | 0.0428 | 0.3039 | 0.2617 | **0.0158** | 186.6 | 357.4 MB |
| SFT | **0.2958** | **0.1627** | **0.0846** | **0.0435** | **0.3335** | **0.2715** | 0.0141 | **191.4** | 358.2 MB |

## Visual-dependence controls

| Checkpoint | Control | BLEU-4 | ROUGE-L | Output change | Token Jaccard distance |
|---|---|---:|---:|---:|---:|
| Pretrain | black | 0.0034 | 0.1962 | 100% | 0.7983 |
| Pretrain | shuffled | 0.0077 | 0.2066 | 100% | 0.7694 |
| SFT | black | 0.0000 | 0.1564 | 100% | 0.8625 |
| SFT | shuffled | 0.0062 | 0.2075 | 100% | 0.8053 |

SFT improves BLEU-1/2/3/4, METEOR-exact, and ROUGE-L over Pretrain on this controlled sample, while the internal CIDEr-style score is slightly lower. The strong degradation under black and shuffled controls shows that the generated captions depend on the supplied image rather than remaining unchanged under visual corruption.

The metric implementation is self-contained so the benchmark can run without Java or COCOEvalCap. Results are suitable for controlled checkpoint comparisons in this repository, but they are not drop-in replacements for official COCOEvalCap paper numbers.

Pilot raw summaries:

- [`coco500_pretrain_vlm_summary.json`](coco500_pretrain_vlm_summary.json)
- [`coco500_sft_vlm_summary.json`](coco500_sft_vlm_summary.json)
