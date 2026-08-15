#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/chain
LOG=logs/chain/dpo_v6_chain.log
log() { echo "$(date '+%F %T') $1" | tee -a "$LOG"; }

log "DPO_V6_START"

if [[ ! -f dataset/dpo_v6_combined.parquet ]]; then
  log "BUILD_BALANCED_PAIRS"
  "$PYTHON_BIN" -u scripts/build_balanced_dpo_pairs.py \
    --vqav2_max 8000 --seed 42 \
    > logs/chain/build_v6_pairs.log 2>&1
  log "BUILD_BALANCED_PAIRS_DONE"
fi

if [[ ! -d out/qwen7b_dpo_v6 ]]; then
  log "DPO_V6"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_BIN" -u scripts/train_qwen_vl_dpo.py \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_knowledge_sft \
    --data_path dataset/dpo_v6_combined.parquet \
    --output_dir out/qwen7b_dpo_v6 \
    --max_samples 0 --epochs 1 --batch_size 1 --grad_accum 8 \
    --lr 5e-6 --beta 0.1 --kl_lambda 0.1 --lora_r 32 --lora_alpha 64 --max_steps 200 \
    --log_interval 10 --seed 42 \
    > logs/chain/qwen7b_dpo_v6.log 2>&1
  log "DPO_V6_DONE"
else
  log "DPO_V6_SKIP"
fi

if [[ ! -f results/okvqa/qwen7b_dpo_v6/summary.json ]]; then
  "$PYTHON_BIN" -u scripts/eval_okvqa.py --model qwen3b \
    --questions_file dataset/okvqa/OpenEnded_mscoco_val2014_questions.json \
    --annotations_file dataset/okvqa/mscoco_val2014_annotations.json \
    --image_zip /autodl-pub/data/COCO14/val2014.zip \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v6 \
    --max_samples 1000 --tag qwen7b_dpo_v6 --output_dir results/okvqa \
    > logs/chain/okvqa_v6.log 2>&1
fi
if [[ ! -f results/mmbench/qwen7b_dpo_v6/summary.json ]]; then
  "$PYTHON_BIN" -u scripts/eval_mmbench.py --dataset_dir dataset/mmbench_en_dev --model qwen3b \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v6 \
    --tag qwen7b_dpo_v6 --output_dir results/mmbench > logs/chain/mmbench_v6.log 2>&1
fi
if [[ ! -f results/official_coco_qwen/qwen7b_dpo_v6/summary.json ]]; then
  "$PYTHON_BIN" -u scripts/eval_qwen_coco.py \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v6 \
    --annotation_file dataset/coco2017/annotations/captions_val2017.json \
    --image_dir dataset/coco2017/val2017 --max_samples 2000 --seed 42 \
    --tag qwen7b_dpo_v6 --output_dir results/official_coco_qwen \
    > logs/chain/coco_v6.log 2>&1
fi
log "DPO_V6_COMPLETE"
