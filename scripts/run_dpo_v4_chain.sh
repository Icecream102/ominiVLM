#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/chain
LOG=logs/chain/dpo_v4_chain.log
log() { echo "$(date '+%F %T') $1" | tee -a "$LOG"; }

while ps -p "$(cat logs/chain/coco_zeroshot.pid 2>/dev/null)" >/dev/null 2>&1; do sleep 60; done
log "DPO_V4_START"

if [[ ! -f dataset/pref_judge_8k.parquet ]]; then
  log "BUILD_JUDGE_8K"
  "$PYTHON_BIN" -u scripts/build_preference_pairs.py \
    --data_path dataset/multitask_sft.parquet \
    --policy_path model/qwen25vl-7b-instruct \
    --policy_adapter out/qwen7b_knowledge_sft \
    --judge_path model/qwen25vl-3b-instruct \
    --max_samples 8000 --gap_threshold 1.0 \
    --temperature 1.2 --top_p 0.9 --max_new_tokens 48 \
    --seed 42 --output dataset/pref_judge_8k.parquet \
    > logs/chain/build_pref_8k.log 2>&1
  log "BUILD_JUDGE_8K_DONE"
else
  log "BUILD_JUDGE_8K_SKIP"
fi

if [[ ! -f dataset/dpo_combined_v4.parquet ]]; then
  log "CONCAT_V4"
  "$PYTHON_BIN" scripts/concat_parquets.py \
    --inputs dataset/okvqa_pairs.parquet dataset/pref_judge_8k.parquet \
    --output dataset/dpo_combined_v4.parquet \
    > logs/chain/concat_v4.log 2>&1
  log "CONCAT_V4_DONE"
fi

if [[ ! -d out/qwen7b_dpo_v4 ]]; then
  log "DPO_V4"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_BIN" -u scripts/train_qwen_vl_dpo.py \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_knowledge_sft \
    --data_path dataset/dpo_combined_v4.parquet \
    --output_dir out/qwen7b_dpo_v4 \
    --max_samples 0 --epochs 1 --batch_size 1 --grad_accum 8 \
    --lr 1e-5 --beta 0.03 --lora_r 32 --lora_alpha 64 --max_steps 300 \
    --log_interval 10 --seed 42 \
    > logs/chain/qwen7b_dpo_v4.log 2>&1
  log "DPO_V4_DONE"
else
  log "DPO_V4_SKIP"
fi

if [[ ! -f results/okvqa/qwen7b_dpo_v4/summary.json ]]; then
  log "EVAL_OKVQA_V4"
  "$PYTHON_BIN" -u scripts/eval_okvqa.py --model qwen3b \
    --questions_file dataset/okvqa/OpenEnded_mscoco_val2014_questions.json \
    --annotations_file dataset/okvqa/mscoco_val2014_annotations.json \
    --image_zip /autodl-pub/data/COCO14/val2014.zip \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v4 \
    --max_samples 1000 --tag qwen7b_dpo_v4 --output_dir results/okvqa \
    > logs/chain/okvqa_v4.log 2>&1
fi
if [[ ! -f results/mmbench/qwen7b_dpo_v4/summary.json ]]; then
  log "EVAL_MMBENCH_V4"
  "$PYTHON_BIN" -u scripts/eval_mmbench.py --dataset_dir dataset/mmbench_en_dev --model qwen3b \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v4 \
    --tag qwen7b_dpo_v4 --output_dir results/mmbench > logs/chain/mmbench_v4.log 2>&1
fi
if [[ ! -f results/official_coco_qwen/qwen7b_dpo_v4/summary.json ]]; then
  log "EVAL_COCO_V4"
  "$PYTHON_BIN" -u scripts/eval_qwen_coco.py \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v4 \
    --annotation_file dataset/coco2017/annotations/captions_val2017.json \
    --image_dir dataset/coco2017/val2017 --max_samples 2000 --seed 42 \
    --tag qwen7b_dpo_v4 --output_dir results/official_coco_qwen \
    > logs/chain/coco_v4.log 2>&1
fi
log "DPO_V4_COMPLETE"
