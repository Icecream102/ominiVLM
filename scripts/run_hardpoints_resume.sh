#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/chain
LOG=logs/chain/hardpoints_resume.log
log() { echo "$(date '+%F %T') $1" | tee -a "$LOG"; }

log "RESUME_START"

if [[ ! -d out/qwen7b_knowledge_sft ]]; then
  log "S2_KNOWLEDGE_SFT"
  "$PYTHON_BIN" -u scripts/train_qwen_vl_qlora.py \
    --data_paths dataset/multitask_sft.parquet dataset/okvqa_sft.parquet \
    --max_samples 9009 --epochs 0.5 --batch_size 2 --grad_accum 16 \
    --lr 2e-5 --lora_r 32 --lora_alpha 64 --max_pixels 401408 \
    --output_dir out/qwen7b_knowledge_sft \
    --save_steps 250 --logging_steps 20 \
    > logs/chain/qwen7b_knowledge_sft.log 2>&1
  log "S2_DONE"
else
  log "S2_SKIP"
fi

if [[ ! -f results/okvqa/qwen7b_knowledge_sft/summary.json ]]; then
  log "EVAL_S2_OKVQA"
  "$PYTHON_BIN" -u scripts/eval_okvqa.py --model qwen3b \
    --questions_file dataset/okvqa/OpenEnded_mscoco_val2014_questions.json \
    --annotations_file dataset/okvqa/mscoco_val2014_annotations.json \
    --image_zip /autodl-pub/data/COCO14/val2014.zip \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_knowledge_sft \
    --max_samples 1000 --tag qwen7b_knowledge_sft --output_dir results/okvqa \
    > logs/chain/okvqa_s2.log 2>&1
fi
if [[ ! -f results/mmbench/qwen7b_knowledge_sft/summary.json ]]; then
  log "EVAL_S2_MMBENCH"
  "$PYTHON_BIN" -u scripts/eval_mmbench.py --dataset_dir dataset/mmbench_en_dev --model qwen3b \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_knowledge_sft \
    --tag qwen7b_knowledge_sft --output_dir results/mmbench > logs/chain/mmbench_s2.log 2>&1
fi
log "EVAL_S2_DONE"

if [[ ! -d out/qwen7b_dpo_v2 ]]; then
  log "S3_DPO_V2"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_BIN" -u scripts/train_qwen_vl_dpo.py \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_knowledge_sft \
    --data_path dataset/okvqa_pairs.parquet \
    --output_dir out/qwen7b_dpo_v2 \
    --max_samples 0 --epochs 1 --batch_size 1 --grad_accum 8 \
    --lr 1e-5 --beta 0.05 --lora_r 32 --lora_alpha 64 --max_steps 200 \
    --log_interval 10 --seed 42 \
    > logs/chain/qwen7b_dpo_v2.log 2>&1
  log "S3_DONE"
else
  log "S3_SKIP"
fi

if [[ ! -f results/okvqa/qwen7b_dpo_v2/summary.json ]]; then
  log "EVAL_S3_OKVQA"
  "$PYTHON_BIN" -u scripts/eval_okvqa.py --model qwen3b \
    --questions_file dataset/okvqa/OpenEnded_mscoco_val2014_questions.json \
    --annotations_file dataset/okvqa/mscoco_val2014_annotations.json \
    --image_zip /autodl-pub/data/COCO14/val2014.zip \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v2 \
    --max_samples 1000 --tag qwen7b_dpo_v2 --output_dir results/okvqa \
    > logs/chain/okvqa_s3.log 2>&1
fi
if [[ ! -f results/mmbench/qwen7b_dpo_v2/summary.json ]]; then
  log "EVAL_S3_MMBENCH"
  "$PYTHON_BIN" -u scripts/eval_mmbench.py --dataset_dir dataset/mmbench_en_dev --model qwen3b \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v2 \
    --tag qwen7b_dpo_v2 --output_dir results/mmbench > logs/chain/mmbench_s3.log 2>&1
fi
log "RESUME_COMPLETE"
