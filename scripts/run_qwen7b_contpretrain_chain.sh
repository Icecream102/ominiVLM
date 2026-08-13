#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/chain
LOG=logs/chain/contpretrain_chain.log
log() { echo "$(date '+%F %T') $1" | tee -a "$LOG"; }

log "CHAIN_START"

if [[ ! -d out/qwen7b_contpretrain ]]; then
  log "STAGE1_CONT_PRETRAIN"
  "$PYTHON_BIN" -u scripts/train_qwen_vl_qlora.py \
    --data_paths dataset/pretrain_i2t.parquet \
    --max_samples 40000 --epochs 0.1 --batch_size 2 --grad_accum 16 \
    --lr 1e-5 --lora_r 64 --lora_alpha 128 --max_pixels 602112 \
    --output_dir out/qwen7b_contpretrain \
    --save_steps 200 --logging_steps 20 \
    > logs/chain/qwen7b_contpretrain.log 2>&1
  log "STAGE1_DONE"
else
  log "STAGE1_SKIP"
fi

log "EVAL_BASE_7B_ZEROSHOT"
"$PYTHON_BIN" -u scripts/eval_mmbench.py \
  --dataset_dir dataset/mmbench_en_dev --model qwen3b \
  --model_path model/qwen25vl-7b-instruct --adapter_path "" \
  --tag qwen7b_zeroshot --output_dir results/mmbench \
  > logs/chain/mmbench_7b_zeroshot.log 2>&1
log "EVAL_CONT_PRETRAIN"
"$PYTHON_BIN" -u scripts/eval_mmbench.py \
  --dataset_dir dataset/mmbench_en_dev --model qwen3b \
  --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_contpretrain \
  --tag qwen7b_contpretrain --output_dir results/mmbench \
  > logs/chain/mmbench_7b_contpretrain.log 2>&1
log "EVAL_CONT_PRETRAIN_DONE"

if [[ ! -d out/qwen7b_contpretrain_dpo ]]; then
  log "STAGE3_DPO_ON_CONT_PRETRAIN"
  "$PYTHON_BIN" -u scripts/train_qwen_vl_dpo.py \
    --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_contpretrain \
    --data_path dataset/preference_pairs.parquet \
    --output_dir out/qwen7b_contpretrain_dpo \
    --max_samples 0 --epochs 1 --batch_size 1 --grad_accum 8 \
    --lr 2e-5 --beta 0.1 --lora_r 32 --lora_alpha 64 --max_steps 100 \
    --log_interval 10 --seed 42 \
    > logs/chain/qwen7b_contpretrain_dpo.log 2>&1
  log "STAGE3_DONE"
else
  log "STAGE3_SKIP"
fi

log "EVAL_CONT_PRETRAIN_DPO"
"$PYTHON_BIN" -u scripts/eval_mmbench.py \
  --dataset_dir dataset/mmbench_en_dev --model qwen3b \
  --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_contpretrain_dpo \
  --tag qwen7b_contpretrain_dpo --output_dir results/mmbench \
  > logs/chain/mmbench_7b_contpretrain_dpo.log 2>&1
log "CHAIN_COMPLETE"
