#!/usr/bin/env bash
# Repair chain: from-scratch VLM (batch 64 + accum), then DPO (r=32 matching
# the 7B QLoRA adapter) on the already-built preference pairs, then DPO evals,
# then METEOR rescore for the 7B QLoRA COCO run.
set -u

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/chain
LOG=logs/chain/repair_chain.log
log() { echo "$(date '+%F %T') $1" | tee -a "$LOG"; }

log "REPAIR_FROM_SCRATCH"
bash scripts/run_from_scratch_vlm.sh > logs/chain/from_scratch_vlm.log 2>&1
log "REPAIR_FROM_SCRATCH_DONE"

log "REPAIR_DPO"
"$PYTHON_BIN" -u scripts/train_qwen_vl_dpo.py \
  --model_path model/qwen25vl-7b-instruct \
  --adapter_path out/qwen7b_qlora_multitask \
  --data_path dataset/preference_pairs.parquet \
  --max_steps 300 --batch_size 1 --grad_accum 8 --lr 2e-5 \
  --lora_r 32 --lora_alpha 64 --beta 0.1 \
  --output_dir out/qwen7b_dpo \
  > logs/chain/qwen7b_dpo.log 2>&1
log "REPAIR_DPO_DONE"

log "REPAIR_DPO_EVALS"
"$PYTHON_BIN" -u scripts/eval_qwen_vl_vqa.py \
  --model_path model/qwen25vl-7b-instruct \
  --adapter_path out/qwen7b_dpo \
  --questions_file dataset/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
  --annotations_file dataset/vqav2/v2_mscoco_val2014_annotations.json \
  --image_zip /autodl-pub/data/COCO14/val2014.zip \
  --max_samples 2000 --output_dir results/vqa_qwen7b_dpo \
  > logs/chain/qwen7b_dpo_vqa.log 2>&1
"$PYTHON_BIN" -u scripts/eval_qwen_coco.py \
  --model_path model/qwen25vl-7b-instruct \
  --adapter_path out/qwen7b_dpo \
  --annotation_file dataset/coco2017/annotations/captions_val2017.json \
  --image_dir dataset/coco2017/val2017 \
  --output_dir results/official_coco_qwen7b_dpo \
  > logs/chain/qwen7b_dpo_coco.log 2>&1
"$PYTHON_BIN" -u scripts/eval_pope.py --model qwen3b \
  --model_path model/qwen25vl-7b-instruct \
  --adapter_path out/qwen7b_dpo \
  --instances_file dataset/coco2017/annotations/instances_val2017.json \
  --image_dir dataset/coco2017/val2017 \
  --output_dir results/pope --tag qwen7b-dpo --constrained \
  > logs/chain/qwen7b_dpo_pope.log 2>&1
log "REPAIR_DPO_EVALS_DONE"

log "REPAIR_METEOR"
"$PYTHON_BIN" -u scripts/rescore_meteor.py \
  --results_root results --weights official_coco_qwen7b official_coco_qwen7b_dpo \
  --meteor_timeout 900 >> logs/chain/meteor_rescore.log 2>&1
log "REPAIR_ALL_DONE"
