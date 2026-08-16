# omniVLM — Lightweight Multimodal LLM Training & Alignment

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange)
![Precision](https://img.shields.io/badge/Precision-bf16-brightgreen)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FQLoRA-lightblue)
![Align](https://img.shields.io/badge/Align-GRPO%2FDPO-purple)
![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey)
</div>

## About

**omniVLM** is a multimodal VLM training project, covering **pretraining, SFT, GRPO, DPO, and evaluation**. 

The project spans training a 65M [MiniMind-V](https://github.com/jingyaogong/minimind-v) model from scratch to continued pretraining and LoRA/QLoRA fine-tuning of Qwen2.5-VL-3B/7B, with hands-on debugging of **GRPO reward misalignment, VQA output mismatch, DPO collapse, and hallucination in small VLMs**. Results are evaluated under COCOEvalCap, VQAv2, OK-VQA, MMBench, and POPE protocols.

**Environment**

**Hardware**: a single RTX 4090 / 4090D (24 GB); bf16 precision.

```bash
# 1.Start-up
conda create -n omini-vlm python=3.12 -y
conda activate omini-vlm
pip install -r requirements.txt

# 2. Build data (multitask SFT: caption + VQAv2 + OK-VQA + MMBench)
python scripts/build_multitask_sft.py

# 3. Training: Pretrain → SFT → multitask SFT 
bash scripts/run_full_training_pipeline.sh

# 4. Evaluation: COCOEvalCap (full val2017) / VQAv2 / OK-VQA / MMBench / POPE
bash scripts/run_official_evals.sh
```

Qwen2.5-VL LoRA / QLoRA / DPO training is provided by `scripts/train_qwen_vl_lora.py`, `scripts/train_qwen_vl_qlora.py`, and `scripts/train_qwen_vl_dpo.py`

## Datasets

| Dataset | Used for                                                                       | Size | Source |
|---|--------------------------------------------------------------------------------|---:|---|
| ALLaVA-4V | Pretrain / SFT image–text alignment (bilingual)                                | Pretrain 1.27M / SFT 2.9M | [HuggingFace · FreedomIntelligence/ALLaVA-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) |
| COCO2017 | Caption training + official COCOEvalCap eval (full val2017, 5,000 images)      | 118k train / 5k val | [cocodataset.org](https://cocodataset.org) |
| VQAv2 | VQA format-alignment training + eval (2,000-question subset, official 3-of-10) | 20k train / 405k val | [visualqa.org](https://visualqa.org) |
| OK-VQA | Open-domain knowledge QA SFT / DPO preference / eval                           | 9,009 train / 5,046 val | [okvqa.allenai.org](https://okvqa.allenai.org) · [lmms-lab/OK-VQA](https://huggingface.co/datasets/lmms-lab/OK-VQA) |
| MMBench en/dev | Comprehensive perception–reasoning benchmark (full 4,329 questions)            | 4,329 | [lmms-lab/MMBench](https://huggingface.co/datasets/lmms-lab/MMBench) |
| POPE | Hallucination eval (positive / random / popular / adversarial)                 | COCO 500 images × 6 questions | In-house, following the [POPE protocol](https://arxiv.org/abs/2306.10378) on COCO2017 annotations |
| Synthetic hallucination data | Hallucination negative samples / preference pairs                              | 1.8 GB+ | In-house (COCO object-detection annotations) |
| DPO preference pairs | GRPO / DPO alignment                                                           | 11,093 pairs (fixed version) | In-house (Qwen2.5-VL-3B judge sampling + OK-VQA annotations) |
| Text-only refusal | Text-robustness control                                                        | — | In-house |

## Models

All final trained checkpoints are published on Hugging Face:
[Luanneee/ominiVLM](https://huggingface.co/Luanneee/ominiVLM).
Base LLMs are loaded from `Qwen/Qwen2.5-VL-3B-Instruct` and `Qwen/Qwen2.5-VL-7B-Instruct`;
the 65M base is trained from scratch.

| Model | Description                                                                              |
| :--- |:-----------------------------------------------------------------------------------------|
| minimind-v-65m-multitask | 65M from-scratch Pretrain → multitask SFT |
| qwen2.5-vl-3b-lora-vqa | 3B LoRA VQA SFT|
| qwen2.5-vl-7b-qlora-multitask | 7B QLoRA multitask|
| qwen2.5-vl-7b-knowledge-sft | 7B knowledge SFT on OK-VQA + multitask  |
| qwen2.5-vl-7b-dpo-v6 | 7B DPO v6 on 11,093 balanced preference pairs |
| qwen2.5-vl-7b-halluc-dpo | 7B hallucination DPO, 400 steps |
| omniVLM-checkpoints | all checkpoints, ablations & datasets |

## Results

### 1. Model scales

| Model |                 Training | VQAv2 | MMBench | OK-VQA | COCO CIDEr | COCO BLEU-4 | POPE |
|---|-------------------------:|---:|---:|---:|---:|---:|---:|
| 65M | Pretrain → multitask SFT | 32.8% | 26.0% | 3.2% | 0.6395 | 0.2271 | 37.4% |
| Qwen2.5-VL-3B |             LoRA| 82.0% | 84.9% | 38.5% | 0.8364 | 0.2566 | 94.2% |
| **Qwen2.5-VL-7B** |      **QLoRA multitask** | **82.9%** | **87.6%** | 48.6%* | **0.9884** | **0.3272** | **95.1%** |

*OK-VQA 48.6% comes from the 7B knowledge SFT

### 2. GRPO reward-alignment (COCOEvalCap)

| Checkpoint | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr | METEOR |
|---|---:|---:|---:|---:|---:|
| Pretrain (1.27M, projection only) | 0.2311 | 0.0250 | 0.2160 | 0.0063 | 0.1483 |
| SFT (2.9M) | 0.2367 | 0.0237 | 0.2136 | 0.0058 | 0.1491 |
| GRPO baseline (group=4, β=0.02) | 0.1561 | 0.0067 | 0.1517 | 0.0002 | 0.1023 |
| GRPO (β=0.10) | 0.2047 | 0.0136 | 0.1883 | 0.0011 | 0.1327 |
| GRPO (adaptive KL) | 0.2235 | 0.0175 | 0.2007 | 0.0013 | 0.1444 |
| GRPO (adaptive KL, group=8) | 0.2262 | 0.0201 | 0.2016 | 0.0016 | 0.1438 |
| **GRPO (CIDEr-aligned reward, group=16)** | **0.2392** | **0.0266** | **0.2159** | 0.0036 | **0.1529** |


### 3. 7B continued pretraining → knowledge SFT → DPO

| Stage | OK-VQA | MMBench | COCO CIDEr |
|---|---:|---:|---:|
| 7B zero-shot | 42.6% | 87.34% | — |
| Caption-only continued pretraining (r=128, 160k) | 39.7% | 87.36% | 0.011 |
| S4 knowledge-mixed continued pretraining | 47.6% | 87.53% | 0.959 |
| Knowledge SFT (OK-VQA 9,009 + multitask 9,009) | **48.6%** | 87.50% | 0.9965 |
| DPO (expanded preferences, β=0.03) | 45.9% | 87.39% | 0.9547 |
| **DPO (balanced preferences, β=0.1)** | 47.7% | 87.32% | **0.9973** |

### 4. VQA–caption data-ratio ablation (65M)

| Recipe                   | VQAv2 acc | COCO BLEU-4 | COCO CIDEr |
|--------------------------|---:|---:|---:|
| SFT baseline (caption)   | 0.0% | 0.0237 | 0.0058 |
| VQA-only SFT (20k)       | **31.6%** | 0.0 (forgotten) | — |
| Mix 1:3                  | 8.4% | — | — |
| Mix 1:1                  | 20.9% | 0.0241 | 0.0067 |
| Mix 3:1                  | 23.0% | — | — |
| **Multitask SFT (231k)** | **32.8%** | **0.2271** | **0.6395** |

### 5. Hallucination control (POPE, COCO 500 images × 6 questions)

| Model                                     | Positive F1 | Random-negative yes rate |
|-------------------------------------------|---:|---:|
| 65M multitask (baseline)                  | 0.878 | 78.7% |
| 65M + hallucination SFT (1:3)             | 0.422 | 6.0% |
| 65M + hallucination SFT (1:1 balanced) | 0.709 | 25.9% |
| Qwen2.5-VL-3B LoRA                        | 0.926 | 2.1% |
| Qwen2.5-VL-7B QLoRA                       | 0.949 | 1.9% |
| 7B + general DPO                          | 0.952 | 2.0% |
| **7B + hallucination DPO (400 steps)**    | **0.957** | 2.6% |

## Cases

### 1. Per-stage cases

|                                          VQA Case 1                                        |                                         VQA Case 2                                         |
|:---------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------:|
| <img src="benchmark_results/official_coco_20260812/samples/single/stage_vqa_145369.png" width="320" alt="vqa_145369"> | <img src="benchmark_results/official_coco_20260812/samples/single/stage_vqa_93852.png" width="320" alt="vqa_93852"> |
|           **Q**: How many elephants are in the photo?<br>**65M multitask**: 1<br>**Qwen2.5-VL-7B LoRA**: 7            |    **Q**: What might the owner's favorite color be?<br>**65M multitask**: blue<br>**Qwen2.5-VL-7B LoRA**: orange    |

|                                                     OK-VQA Case 1                                                       | OK-VQA Case 2 |
|:-----------------------------------------------------------------------------------------------------------------------:| :---: |
| <img src="benchmark_results/official_coco_20260812/samples/single/stage_okvqa_53420.png" width="320" alt="okvqa_53420"> | <img src="benchmark_results/official_coco_20260812/samples/single/stage_okvqa_303026.png" width="320" alt="okvqa_303026"> |
|      **Q**: Which country is this sport big in?<br>**zero-shot**: USA<br>**knowledge SFT**: usa<br>**DPO (balanced pairs)**: usa      | **Q**: What is the woman sitting on?<br>**zero-shot**: bench<br>**knowledge SFT**: bench<br>**DPO (balanced pairs)**: bench |

| DPO Case 1 |  DPO Case 2 |
| :---: | :---: |
| <img src="benchmark_results/official_coco_20260812/samples/single/stage_dpo_53420.png" width="320" alt="dpo_53420"> | <img src="benchmark_results/official_coco_20260812/samples/single/stage_dpo_303026.png" width="320" alt="dpo_303026"> |
| **Q**: Which country is this sport big in?<br>**DPO (VQAv2-only pairs, β=0.02)**: usa addCriterion<br>**DPO (balanced pairs, β=0.1)**: usa | **Q**: What is the woman sitting on?<br>**DPO (VQAv2-only pairs, β=0.02)**: bench addCriterion<br>**DPO (balanced pairs, β=0.1)**: bench |

### 2. Visual-dependence control (SFT model; GT / original input / all-black / mismatched)

| Case | GT (reference caption) | Original input | All-black input | Mismatched input |
|:------:| :---: | :---: | :---: | :---: |
| Case 1 | <img src="benchmark_results/official_coco_20260812/samples/single/vd_530466_original.png" width="140" height="140" alt="vd530466_gt"><br><small>A passenger train that has some graffiti on it.</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_530466_original.png" width="140" height="140" alt="vd530466_correct"><br><small>The image depicts a vibrant and colorful train track…</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_530466_black.png" width="140" height="140" alt="vd530466_black"><br><small>The image is a simple, unadorned black background…</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_530466_shuffled.png" width="140" height="140" alt="vd530466_shuffled"><br><small>(input 233771) The image is a black and white photograph of a cityscape…</small> |
| Case 2 | <img src="benchmark_results/official_coco_20260812/samples/single/vd_233771_original.png" width="140" height="140" alt="vd233771_gt"><br><small>A black and white image with a colored british flag umbrella.</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_233771_original.png" width="140" height="140" alt="vd233771_correct"><br><small>The image is a black and white photograph of a cityscape…</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_233771_black.png" width="140" height="140" alt="vd233771_black"><br><small>The image is a simple, unadorned black background…</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_233771_shuffled.png" width="140" height="140" alt="vd233771_shuffled"><br><small>(input 475572) The image depicts a collection of three-dimensional objects…</small> |
| Case 3 | <img src="benchmark_results/official_coco_20260812/samples/single/vd_475572_original.png" width="140" height="140" alt="vd475572_gt"><br><small>A Beanie Baby beside a vintage photo of a man and a woman.</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_475572_original.png" width="140" height="140" alt="vd475572_correct"><br><small>The image depicts a collection of three-dimensional objects…</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_475572_black.png" width="140" height="140" alt="vd475572_black"><br><small>The image is a simple, unadorned black background…</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_475572_shuffled.png" width="140" height="140" alt="vd475572_shuffled"><br><small>(input 89697) The image is a monochromatic photograph of a man sitting on a bench…</small> |
| Case 4 | <img src="benchmark_results/official_coco_20260812/samples/single/vd_89697_original.png" width="140" height="140" alt="vd89697_gt"><br><small>The man is sitting on the arm of a bench near a woman.</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_89697_original.png" width="140" height="140" alt="vd89697_correct"><br><small>The image is a monochromatic photograph of a man sitting on a bench…</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_89697_black.png" width="140" height="140" alt="vd89697_black"><br><small>The image is a simple, unadorned black background…</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_89697_shuffled.png" width="140" height="140" alt="vd89697_shuffled"><br><small>(input 109313) The image depicts a person in a room that appears to be a woman…</small> |
| Case 5 | <img src="benchmark_results/official_coco_20260812/samples/single/vd_109313_original.png" width="140" height="140" alt="vd109313_gt"><br><small>A man holding a tv remote and wii controller.</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_109313_original.png" width="140" height="140" alt="vd109313_correct"><br><small>The image depicts a person in a room that appears to be a woman…</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_109313_black.png" width="140" height="140" alt="vd109313_black"><br><small>The image is a simple, unadorned black background…</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_109313_shuffled.png" width="140" height="140" alt="vd109313_shuffled"><br><small>(input 579893) The image depicts a serene, serene landscape during daytime…</small> |
| Case 6 | <img src="benchmark_results/official_coco_20260812/samples/single/vd_579893_original.png" width="140" height="140" alt="vd579893_gt"><br><small>A close up of the stop sign and to street signs.</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_579893_original.png" width="140" height="140" alt="vd579893_correct"><br><small>The image depicts a serene, serene landscape during daytime…</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_579893_black.png" width="140" height="140" alt="vd579893_black"><br><small>The image is a simple, unadorned black background…</small> | <img src="benchmark_results/official_coco_20260812/samples/single/vd_579893_shuffled.png" width="140" height="140" alt="vd579893_shuffled"><br><small>(input 530466) The image depicts a vibrant and colorful train track…</small> |

## Repository Structure

```text
omniVLM/
├── model/                          # Model definitions
│   ├── model_vlm.py                # MiniMind-V VLM (SigLIP2 + MLP projection + LLM)
│   ├── model_minimind.py           # MiniMind language backbone
│   └── torch_compat.py             # Legacy torch compatibility layer
├── dataset/                        # Data construction and loading
│   ├── lm_dataset.py               # LM / image-text SFT data loading
│   ├── grpo_dataset.py             # GRPO sampling dataset
│   └── *.parquet                   # pretrain_i2t / sft_i2t / multitask_sft / dpo_v6_combined ...
├── trainer/                        # Trainers
│   ├── train_pretrain_vlm.py       # Pretrain (projection only)
│   ├── train_sft_vlm.py            # SFT (frozen / first-last unfreeze / multitask)
│   ├── train_grpo_vlm.py           # GRPO (adaptive KL, combined reward, safety stop)
│   └── trainer_utils.py            # Checkpoint resume, config validation, .done markers
├── scripts/                        # One-command experiment chains and tools
│   ├── build_*.py                  # Data builders: multitask / OK-VQA / hallucination / preference pairs / OCR / spatial QA
│   ├── train_qwen_vl_lora.py       # Qwen2.5-VL LoRA training
│   ├── train_qwen_vl_qlora.py      # Qwen2.5-VL QLoRA training
│   ├── train_qwen_vl_dpo.py        # Qwen2.5-VL DPO training
│   ├── eval_coco_official.py       # Official COCOEvalCap (with METEOR compatibility fix)
│   ├── eval_vqa.py                 # VQAv2 official 3-of-10 evaluation
│   ├── eval_okvqa.py / eval_mmbench.py / eval_pope.py / eval_text_only.py
│   ├── run_*.sh                    # Idempotent pipelines: full chain / GRPO fix / DPO chain / eval
│   └── analyze_*.py                # Data profiling / judge reward correlation / DPO effect analysis
├── evaluation/                     # Metrics and rewards
│   ├── caption_metrics.py          # BLEU / ROUGE / CIDEr / METEOR-exact
│   └── grpo_rewards.py             # Configurable combined rewards (unigram / ROUGE / METEOR / CIDEr / length / repetition)
├── configs/                        # Experiment configs (JSON)
├── benchmark_results/              # Official eval results, charts, per-stage sample visualizations
├── docs/                           # Experiment reports / results summary / resume entries
└── tests/                          # Unit tests
```

## Acknowledgements

- Model architecture and training framework are based on [MiniMind-V](https://github.com/jingyaogong/minimind-v) 
- Pretraining / SFT data comes from [ALLaVA-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V)
- Evaluation follows the official protocols of [COCOEvalCap](https://github.com/tylin/coco-caption), [MMBench](https://github.com/open-compass/MMBench), and [POPE](https://github.com/RUCAIBox/POPE)