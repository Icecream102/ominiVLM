#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/chain
LOG=logs/chain/dpo_v4b_chain.log
log() { echo "$(date '+%F %T') $1" | tee -a "$LOG"; }

log "DPO_V4B_START"

if [[ ! -f dataset/dpo_combined_v4_shuffled.parquet ]]; then
  log "SHUFFLE_COMBINED"
  "$PYTHON_BIN" -c "
import random
import pyarrow as pa
import pyarrow.parquet as pq
t = pq.read_table('dataset/dpo_combined_v4.parquet')
rng = random.Random(42)
order = rng.sample(range(t.num_rows), t.num_rows)
pq.write_table(t.take(order), 'dataset/dpo_combined_v4_shuffled.parquet', compression='snappy')
print('shuffled rows:', t.num_rows)
" > logs/chain/shuffle_v4.log 2>&1
  log "SHUFFLE_DONE"
fi

if [[ ! -d out/qwen7b_dpo_v4b ]]; then
  log "DPO_V4B"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_BIN" -u scripts/train_qwen_vl_dpo.py \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_knowledge_sft \
    --data_path dataset/dpo_combined_v4_shuffled.parquet \
    --output_dir out/qwen7b_dpo_v4b \
    --max_samples 0 --epochs 1 --batch_size 1 --grad_accum 8 \
    --lr 1e-5 --beta 0.03 --lora_r 32 --lora_alpha 64 --max_steps 300 \
    --log_interval 10 --seed 42 \
    > logs/chain/qwen7b_dpo_v4b.log 2>&1
  log "DPO_V4B_DONE"
else
  log "DPO_V4B_SKIP"
fi

for metric in okvqa mmbench coco; do
  case "$metric" in
    okvqa)
      if [[ ! -f results/okvqa/qwen7b_dpo_v4b/summary.json ]]; then
        "$PYTHON_BIN" -u scripts/eval_okvqa.py --model qwen3b \
          --questions_file dataset/okvqa/OpenEnded_mscoco_val2014_questions.json \
          --annotations_file dataset/okvqa/mscoco_val2014_annotations.json \
          --image_zip /autodl-pub/data/COCO14/val2014.zip \
          --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v4b \
          --max_samples 1000 --tag qwen7b_dpo_v4b --output_dir results/okvqa \
          > logs/chain/okvqa_v4b.log 2>&1
      fi ;;
    mmbench)
      if [[ ! -f results/mmbench/qwen7b_dpo_v4b/summary.json ]]; then
        "$PYTHON_BIN" -u scripts/eval_mmbench.py --dataset_dir dataset/mmbench_en_dev --model qwen3b \
          --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v4b \
          --tag qwen7b_dpo_v4b --output_dir results/mmbench > logs/chain/mmbench_v4b.log 2>&1
      fi ;;
    coco)
      if [[ ! -f results/official_coco_qwen/qwen7b_dpo_v4b/summary.json ]]; then
        "$PYTHON_BIN" -u scripts/eval_qwen_coco.py \
          --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v4b \
          --annotation_file dataset/coco2017/annotations/captions_val2017.json \
          --image_dir dataset/coco2017/val2017 --max_samples 2000 --seed 42 \
          --tag qwen7b_dpo_v4b --output_dir results/official_coco_qwen \
          > logs/chain/coco_v4b.log 2>&1
      fi ;;
  esac
done
log "DPO_V4B_COMPLETE"
