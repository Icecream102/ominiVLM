#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/chain
LOG=logs/chain/hardpoints_chain.log
log() { echo "$(date '+%F %T') $1" | tee -a "$LOG"; }

log "HP_CHAIN_START"

if [[ ! -f dataset/okvqa_sft.parquet ]]; then
  log "BUILD_OKVQA"
  "$PYTHON_BIN" -u scripts/build_okvqa_sft.py > logs/chain/build_okvqa.log 2>&1
  log "BUILD_OKVQA_DONE"
fi

log "EVAL_7B_ZEROSHOT_OKVQA"
"$PYTHON_BIN" -u scripts/eval_okvqa.py --model qwen3b \
  --questions_file dataset/okvqa/OpenEnded_mscoco_val2014_questions.json \
  --annotations_file dataset/okvqa/mscoco_val2014_annotations.json \
  --image_zip /autodl-pub/data/COCO14/val2014.zip \
  --model_path model/qwen25vl-7b-instruct --adapter_path "" \
  --max_samples 1000 --tag qwen7b_zeroshot --output_dir results/okvqa \
  > logs/chain/okvqa_7b_zeroshot.log 2>&1
log "EVAL_7B_ZEROSHOT_OKVQA_DONE"

if [[ ! -d out/qwen7b_contpretrain_v2 ]]; then
  log "S1_CONT_PRETRAIN_V2"
  "$PYTHON_BIN" -u scripts/train_qwen_vl_qlora.py \
    --data_paths dataset/pretrain_i2t.parquet \
    --max_samples 160000 --epochs 0.05 --batch_size 2 --grad_accum 16 \
    --lr 1e-5 --lora_r 128 --lora_alpha 256 --max_pixels 401408 \
    --output_dir out/qwen7b_contpretrain_v2 \
    --save_steps 250 --logging_steps 20 \
    > logs/chain/qwen7b_contpretrain_v2.log 2>&1
  log "S1_DONE"
else
  log "S1_SKIP"
fi

log "EVAL_S1"
"$PYTHON_BIN" -u scripts/eval_mmbench.py --dataset_dir dataset/mmbench_en_dev --model qwen3b \
  --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_contpretrain_v2 \
  --tag qwen7b_contpretrain_v2 --output_dir results/mmbench > logs/chain/mmbench_s1.log 2>&1
"$PYTHON_BIN" -u scripts/eval_okvqa.py --model qwen3b \
  --questions_file dataset/okvqa/OpenEnded_mscoco_val2014_questions.json \
  --annotations_file dataset/okvqa/mscoco_val2014_annotations.json \
  --image_zip /autodl-pub/data/COCO14/val2014.zip \
  --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_contpretrain_v2 \
  --max_samples 1000 --tag qwen7b_contpretrain_v2 --output_dir results/okvqa \
  > logs/chain/okvqa_s1.log 2>&1
log "EVAL_S1_DONE"

if [[ ! -d out/qwen7b_knowledge_sft ]]; then
  log "S2_KNOWLEDGE_SFT"
  "$PYTHON_BIN" -u scripts/train_qwen_vl_qlora.py \
    --data_paths dataset/multitask_sft.parquet dataset/okvqa_sft.parquet \
    --max_samples 12000 9009 --epochs 0.4 --batch_size 2 --grad_accum 16 \
    --lr 2e-5 --lora_r 32 --lora_alpha 64 --max_pixels 401408 \
    --output_dir out/qwen7b_knowledge_sft \
    --save_steps 250 --logging_steps 20 \
    > logs/chain/qwen7b_knowledge_sft.log 2>&1
  log "S2_DONE"
else
  log "S2_SKIP"
fi

log "EVAL_S2"
"$PYTHON_BIN" -u scripts/eval_okvqa.py --model qwen3b \
  --questions_file dataset/okvqa/OpenEnded_mscoco_val2014_questions.json \
  --annotations_file dataset/okvqa/mscoco_val2014_annotations.json \
  --image_zip /autodl-pub/data/COCO14/val2014.zip \
  --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_knowledge_sft \
  --max_samples 1000 --tag qwen7b_knowledge_sft --output_dir results/okvqa \
  > logs/chain/okvqa_s2.log 2>&1
"$PYTHON_BIN" -u scripts/eval_mmbench.py --dataset_dir dataset/mmbench_en_dev --model qwen3b \
  --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_knowledge_sft \
  --tag qwen7b_knowledge_sft --output_dir results/mmbench > logs/chain/mmbench_s2.log 2>&1
log "EVAL_S2_DONE"

if [[ ! -d out/qwen7b_dpo_v2 ]]; then
  log "S3_DPO_V2"
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

log "EVAL_S3"
"$PYTHON_BIN" -u scripts/eval_okvqa.py --model qwen3b \
  --questions_file dataset/okvqa/OpenEnded_mscoco_val2014_questions.json \
  --annotations_file dataset/okvqa/mscoco_val2014_annotations.json \
  --image_zip /autodl-pub/data/COCO14/val2014.zip \
  --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v2 \
  --max_samples 1000 --tag qwen7b_dpo_v2 --output_dir results/okvqa \
  > logs/chain/okvqa_s3.log 2>&1
"$PYTHON_BIN" -u scripts/eval_mmbench.py --dataset_dir dataset/mmbench_en_dev --model qwen3b \
  --model_path model/qwen25vl-7b-instruct --adapter_path out/qwen7b_dpo_v2 \
  --tag qwen7b_dpo_v2 --output_dir results/mmbench > logs/chain/mmbench_s3.log 2>&1
log "HP_CHAIN_COMPLETE"
