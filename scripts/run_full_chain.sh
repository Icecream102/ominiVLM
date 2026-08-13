#!/usr/bin/env bash
# Master autonomous chain (no interactive approvals):
#   smoke 7B QLoRA -> full 7B QLoRA -> 7B evals -> from-scratch VLM
#   -> judge preference pairs -> DPO -> final evals -> sync markers.
set -u

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/chain
LOG=logs/chain/master_chain.log

log() { echo "$(date '+%F %T') $1" | tee -a "$LOG"; }

# Wait for the smoke test launched by the earlier watchdog (marker in master_chain.log).
while ! grep -q QWEN7B_SMOKE_DONE logs/master_chain.log 2>/dev/null; do sleep 60; done
log "SMOKE_OK"

# 1) Full 7B QLoRA multi-task training.
log "START_QWEN7B_FULL"
"$PYTHON_BIN" -u scripts/train_qwen_vl_qlora.py \
  --data_paths dataset/multitask_sft.parquet dataset/hallucination_sft.parquet \
    dataset/spatial_qa.parquet dataset/synthetic_ocr.parquet \
  --max_samples 2000 --epochs 1 --batch_size 1 --grad_accum 8 --lr 2e-4 --lora_r 32 \
  --max_pixels 401408 --output_dir out/qwen7b_qlora_multitask \
  --save_steps 1000 --logging_steps 20 \
  > logs/chain/qwen7b_full.log 2>&1
log "DONE_QWEN7B_FULL"

# 2) Evaluate the 7B QLoRA model.
log "START_QWEN7B_EVALS"
"$PYTHON_BIN" -u scripts/eval_qwen_vl_vqa.py \
  --model_path model/qwen25vl-7b-instruct \
  --adapter_path out/qwen7b_qlora_multitask \
  --questions_file dataset/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
  --annotations_file dataset/vqav2/v2_mscoco_val2014_annotations.json \
  --image_zip /autodl-pub/data/COCO14/val2014.zip \
  --max_samples 2000 --output_dir results/vqa_qwen7b \
  > logs/chain/qwen7b_vqa.log 2>&1
"$PYTHON_BIN" -u scripts/eval_qwen_coco.py \
  --model_path model/qwen25vl-7b-instruct \
  --adapter_path out/qwen7b_qlora_multitask \
  --annotation_file dataset/coco2017/annotations/captions_val2017.json \
  --image_dir dataset/coco2017/val2017 \
  --output_dir results/official_coco_qwen7b \
  > logs/chain/qwen7b_coco.log 2>&1
"$PYTHON_BIN" -u scripts/eval_mmbench.py --model qwen3b \
  --model_path model/qwen25vl-7b-instruct \
  --adapter_path out/qwen7b_qlora_multitask \
  --output_dir results/mmbench --tag qwen7b-qlora \
  > logs/chain/qwen7b_mmbench.log 2>&1
"$PYTHON_BIN" -u scripts/eval_pope.py --model qwen3b \
  --model_path model/qwen25vl-7b-instruct \
  --adapter_path out/qwen7b_qlora_multitask \
  --instances_file dataset/coco2017/annotations/instances_val2017.json \
  --image_dir dataset/coco2017/val2017 \
  --output_dir results/pope --tag qwen7b-qlora --constrained \
  > logs/chain/qwen7b_pope.log 2>&1
log "DONE_QWEN7B_EVALS"

# 3) From-scratch VLM pipeline (unfrozen LLM pretrain -> multitask SFT -> evals).
log "START_FROM_SCRATCH"
bash scripts/run_from_scratch_vlm.sh > logs/chain/from_scratch_vlm.log 2>&1
log "DONE_FROM_SCRATCH"

# 4) Judge preference pairs from the 7B QLoRA policy.
log "START_PREFERENCE"
"$PYTHON_BIN" -u scripts/build_preference_pairs.py \
  --data_path dataset/vqa_sft.parquet \
  --policy_path model/qwen25vl-7b-instruct \
  --policy_adapter out/qwen7b_qlora_multitask \
  --judge_path model/qwen25vl-3b-instruct \
  --judge_adapter out/qwen_vl_lora \
  --max_samples 1500 --gap_threshold 1.0 \
  --output dataset/preference_pairs.parquet \
  > logs/chain/preference.log 2>&1
log "DONE_PREFERENCE"

# 5) DPO on the 7B QLoRA model.
log "START_DPO"
"$PYTHON_BIN" -u scripts/train_qwen_vl_dpo.py \
  --model_path model/qwen25vl-7b-instruct \
  --adapter_path out/qwen7b_qlora_multitask \
  --data_path dataset/preference_pairs.parquet \
  --max_steps 300 --batch_size 1 --grad_accum 8 --lr 2e-5 \
  --beta 0.1 --output_dir out/qwen7b_dpo \
  > logs/chain/qwen7b_dpo.log 2>&1
log "DONE_DPO"

# 6) Final DPO evals (subset).
log "START_DPO_EVALS"
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
log "DONE_DPO_EVALS"

log "ALL_CHAIN_DONE"
