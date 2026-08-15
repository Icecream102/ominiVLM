#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/chain
LOG=logs/chain/contpretrain_positive.log
log() { echo "$(date '+%F %T') $1" | tee -a "$LOG"; }

# Wait for the DPO v3 chain to release the GPU.
while ps -p "$(cat logs/chain/dpo_v3.pid 2>/dev/null)" >/dev/null 2>&1; do sleep 60; done
log "POSITIVE_START"

COCO_ARGS=(--annotation_file dataset/coco2017/annotations/captions_val2017.json
           --image_dir dataset/coco2017/val2017 --max_samples 2000 --seed 42)

eval_coco() {
  local tag="$1" adapter="$2"
  if [[ ! -f "results/official_coco_qwen/$tag/summary.json" ]]; then
    log "COCO_$tag"
    "$PYTHON_BIN" -u scripts/eval_qwen_coco.py \
      --model_path model/qwen25vl-7b-instruct --adapter_path "$adapter" \
      "${COCO_ARGS[@]}" --tag "$tag" --output_dir results/official_coco_qwen \
      > "logs/chain/coco_${tag}.log" 2>&1
  fi
}

eval_coco qwen7b_zeroshot ""
eval_coco qwen7b_contpretrain_v2 out/qwen7b_contpretrain_v2
eval_coco qwen7b_knowledge_sft out/qwen7b_knowledge_sft
log "COCO_EVALS_DONE"

if [[ ! -d out/qwen7b_contpretrain_knowledge ]]; then
  log "S4_KNOWLEDGE_CONT_PRETRAIN"
  "$PYTHON_BIN" -u scripts/train_qwen_vl_qlora.py \
    --data_paths dataset/pretrain_i2t.parquet dataset/okvqa_sft.parquet dataset/multitask_sft.parquet \
    --max_samples 20000 --epochs 0.2 --batch_size 2 --grad_accum 16 \
    --lr 1e-5 --lora_r 64 --lora_alpha 128 --max_pixels 401408 \
    --output_dir out/qwen7b_contpretrain_knowledge \
    --save_steps 250 --logging_steps 20 \
    > logs/chain/qwen7b_contpretrain_knowledge.log 2>&1
  log "S4_DONE"
fi

if [[ ! -f results/okvqa/qwen7b_contpretrain_knowledge/summary.json ]]; then
  log "EVAL_S4_OKVQA"
  "$PYTHON_BIN" -u scripts/eval_okvqa.py --model qwen3b \
    --questions_file dataset/okvqa/OpenEnded_mscoco_val2014_questions.json \
    --annotations_file dataset/okvqa/mscoco_val2014_annotations.json \
    --image_zip /autodl-pub/data/COCO14/val2014.zip \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_contpretrain_knowledge \
    --max_samples 1000 --tag qwen7b_contpretrain_knowledge --output_dir results/okvqa \
    > logs/chain/okvqa_s4.log 2>&1
fi
if [[ ! -f results/mmbench/qwen7b_contpretrain_knowledge/summary.json ]]; then
  log "EVAL_S4_MMBENCH"
  "$PYTHON_BIN" -u scripts/eval_mmbench.py --dataset_dir dataset/mmbench_en_dev --model qwen3b \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_contpretrain_knowledge \
    --tag qwen7b_contpretrain_knowledge --output_dir results/mmbench \
    > logs/chain/mmbench_s4.log 2>&1
fi
eval_coco qwen7b_contpretrain_knowledge out/qwen7b_contpretrain_knowledge
if [[ ! -f results/official_coco_qwen/qwen7b_dpo_v3/summary.json ]]; then
  eval_coco qwen7b_dpo_v3 out/qwen7b_dpo_v3
fi
log "POSITIVE_COMPLETE"
