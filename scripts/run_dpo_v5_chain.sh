#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/chain
LOG=logs/chain/dpo_v5_chain.log
log() { echo "$(date '+%F %T') $1" | tee -a "$LOG"; }

log "DPO_V5_START"

if [[ ! -f dataset/vqav2_pairs.parquet ]]; then
  log "BUILD_VQAV2_PAIRS"
  "$PYTHON_BIN" -u scripts/build_vqav2_pairs.py \
    --max_samples 20000 --seed 42 \
    > logs/chain/build_vqav2_pairs.log 2>&1
  log "BUILD_VQAV2_PAIRS_DONE"
fi

if [[ ! -f dataset/dpo_v5_combined.parquet ]]; then
  log "CONCAT_V5"
  "$PYTHON_BIN" -c "
import random
import pyarrow as pa
import pyarrow.parquet as pq
tables = [pq.read_table(f) for f in ['dataset/okvqa_pairs.parquet', 'dataset/vqav2_pairs.parquet']]
t = pa.concat_tables(tables, promote_options='default')
rng = random.Random(42)
order = rng.sample(range(t.num_rows), t.num_rows)
pq.write_table(t.take(order), 'dataset/dpo_v5_combined.parquet', compression='snappy')
print('combined rows:', t.num_rows)
" > logs/chain/concat_v5.log 2>&1
  log "CONCAT_V5_DONE"
fi

if [[ ! -d out/qwen7b_dpo_v5 ]]; then
  log "DPO_V5"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PYTHON_BIN" -u scripts/train_qwen_vl_dpo.py \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_knowledge_sft \
    --data_path dataset/dpo_v5_combined.parquet \
    --output_dir out/qwen7b_dpo_v5 \
    --max_samples 0 --epochs 1 --batch_size 1 --grad_accum 8 \
    --lr 1e-5 --beta 0.02 --lora_r 32 --lora_alpha 64 --max_steps 400 \
    --log_interval 10 --seed 42 \
    > logs/chain/qwen7b_dpo_v5.log 2>&1
  log "DPO_V5_DONE"
else
  log "DPO_V5_SKIP"
fi

if [[ ! -f results/okvqa/qwen7b_dpo_v5/summary.json ]]; then
  "$PYTHON_BIN" -u scripts/eval_okvqa.py --model qwen3b \
    --questions_file dataset/okvqa/OpenEnded_mscoco_val2014_questions.json \
    --annotations_file dataset/okvqa/mscoco_val2014_annotations.json \
    --image_zip /autodl-pub/data/COCO14/val2014.zip \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v5 \
    --max_samples 1000 --tag qwen7b_dpo_v5 --output_dir results/okvqa \
    > logs/chain/okvqa_v5.log 2>&1
fi
if [[ ! -f results/mmbench/qwen7b_dpo_v5/summary.json ]]; then
  "$PYTHON_BIN" -u scripts/eval_mmbench.py --dataset_dir dataset/mmbench_en_dev --model qwen3b \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v5 \
    --tag qwen7b_dpo_v5 --output_dir results/mmbench > logs/chain/mmbench_v5.log 2>&1
fi
if [[ ! -f results/official_coco_qwen/qwen7b_dpo_v5/summary.json ]]; then
  "$PYTHON_BIN" -u scripts/eval_qwen_coco.py \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v5 \
    --annotation_file dataset/coco2017/annotations/captions_val2017.json \
    --image_dir dataset/coco2017/val2017 --max_samples 2000 --seed 42 \
    --tag qwen7b_dpo_v5 --output_dir results/official_coco_qwen \
    > logs/chain/coco_v5.log 2>&1
fi
log "DPO_V5_COMPLETE"
