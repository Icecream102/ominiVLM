#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-2}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
GRPO_SAMPLES="${GRPO_SAMPLES:-5000}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-128}"
PRETRAIN_LR="${PRETRAIN_LR:-4e-4}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-64}"
SFT_LR="${SFT_LR:-5e-6}"
GRPO_GROUP_SIZE="${GRPO_GROUP_SIZE:-4}"
GRPO_PPO_EPOCHS="${GRPO_PPO_EPOCHS:-2}"
GRPO_LR="${GRPO_LR:-1e-6}"
GRPO_BETA="${GRPO_BETA:-0.02}"
GRPO_ADAPTIVE_BETA="${GRPO_ADAPTIVE_BETA:-1}"
GRPO_TARGET_KL="${GRPO_TARGET_KL:-0.10}"
GRPO_KL_STOP="${GRPO_KL_STOP:-1.0}"
EVAL_SAMPLES="${EVAL_SAMPLES:-500}"
PIPELINE_SEED="${PIPELINE_SEED:-42}"
read -r -a EVAL_CONDITION_ARGS <<< "${EVAL_CONDITIONS:-correct black shuffled}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-tmp/hf-datasets-cache}"

mkdir -p logs checkpoints/full out results/full_pipeline/coco500 "$HF_HOME" "$HF_DATASETS_CACHE"

if [[ ! -f logs/full_pretrain.done ]]; then
  resume=0
  [[ -f checkpoints/full/pretrain_full_vlm_768_resume.pth ]] && resume=1
  "$PYTHON_BIN" -u trainer/train_pretrain_vlm.py \
    --data_path dataset/pretrain_i2t.parquet \
    --save_dir out --checkpoint_dir checkpoints/full \
    --from_weight llm --save_weight pretrain_full_vlm \
    --epochs "$PRETRAIN_EPOCHS" --batch_size "$PRETRAIN_BATCH_SIZE" \
    --learning_rate "$PRETRAIN_LR" --freeze_llm 2 \
    --num_workers 8 --prefetch_factor 4 --persistent_workers 1 \
    --save_interval 10000 --log_interval 100 --from_resume "$resume" --seed "$PIPELINE_SEED" \
    2>&1 | tee -a logs/full_pretrain.log
  touch logs/full_pretrain.done
fi

if [[ ! -f logs/full_sft.done ]]; then
  resume=0
  [[ -f checkpoints/full/sft_full_vlm_768_resume.pth ]] && resume=1
  "$PYTHON_BIN" -u trainer/train_sft_vlm.py \
    --data_path dataset/sft_i2t.parquet \
    --save_dir out --checkpoint_dir checkpoints/full \
    --from_weight pretrain_full_vlm --save_weight sft_full_vlm \
    --epochs "$SFT_EPOCHS" --batch_size "$SFT_BATCH_SIZE" \
    --learning_rate "$SFT_LR" --freeze_llm 1 \
    --num_workers 8 --prefetch_factor 4 --persistent_workers 1 \
    --save_interval 20000 --log_interval 200 --from_resume "$resume" --seed "$PIPELINE_SEED" \
    2>&1 | tee -a logs/full_sft.log
  touch logs/full_sft.done
fi

if [[ ! -f dataset/grpo_i2t.parquet ]]; then
  "$PYTHON_BIN" scripts/build_grpo_dataset.py \
    --input dataset/sft_i2t.parquet \
    --output dataset/grpo_i2t.parquet \
    --samples "$GRPO_SAMPLES" --seed "$PIPELINE_SEED" \
    2>&1 | tee logs/build_grpo_dataset.log
fi

if [[ ! -f logs/full_grpo.done ]]; then
  resume=0
  [[ -f checkpoints/full/grpo_full_vlm_768_resume.pth ]] && resume=1
  "$PYTHON_BIN" -u trainer/train_grpo_vlm.py \
    --data_path dataset/grpo_i2t.parquet \
    --save_dir out --checkpoint_dir checkpoints/full \
    --from_weight sft_full_vlm --save_weight grpo_full_vlm \
    --epochs 1 --group_size "$GRPO_GROUP_SIZE" --ppo_epochs "$GRPO_PPO_EPOCHS" \
    --learning_rate "$GRPO_LR" --beta "$GRPO_BETA" --clip_eps 0.2 \
    --adaptive_beta "$GRPO_ADAPTIVE_BETA" --target_kl "$GRPO_TARGET_KL" \
    --kl_stop "$GRPO_KL_STOP" \
    --max_prompt_length 384 --max_new_tokens 48 \
    --num_workers 4 --save_interval 250 --log_interval 10 \
    --resume "$resume" --log_file logs/full_grpo.jsonl --seed "$PIPELINE_SEED" \
    2>&1 | tee -a logs/full_grpo.log
  touch logs/full_grpo.done
fi

for weight in pretrain_full_vlm sft_full_vlm grpo_full_vlm; do
  if [[ ! -f "results/full_pipeline/coco500/$weight/summary.json" ]]; then
    "$PYTHON_BIN" -u eval_vlm_benchmark.py \
      --annotation_file dataset/coco2017/annotations/captions_val2017.json \
      --image_dir dataset/coco2017/val2017 \
      --weight "$weight" --save_dir out \
      --max_samples "$EVAL_SAMPLES" --max_new_tokens 48 \
      --conditions "${EVAL_CONDITION_ARGS[@]}" --seed "$PIPELINE_SEED" \
      --output_dir results/full_pipeline/coco500 \
      2>&1 | tee "logs/eval_${weight}.log"
  fi
done

touch logs/full_pipeline.done
echo "Full Pretrain -> SFT -> GRPO -> evaluation pipeline complete."
