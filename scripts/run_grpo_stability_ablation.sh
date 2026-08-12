#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
SEED="${SEED:-42}"
DATA_PATH="${DATA_PATH:-dataset/grpo_i2t.parquet}"
COCO_ANNOTATIONS="${COCO_ANNOTATIONS:-dataset/coco2017/annotations/captions_val2017.json}"
COCO_IMAGES="${COCO_IMAGES:-dataset/coco2017/val2017}"

mkdir -p logs/ablations checkpoints/ablations results/grpo_stability/coco500 out

run_grpo() {
  local name="$1"
  shift
  local resume=0
  [[ -f "checkpoints/ablations/${name}_768_resume.pth" ]] && resume=1
  if [[ ! -f "logs/ablations/${name}.done" ]]; then
    "$PYTHON_BIN" -u trainer/train_grpo_vlm.py \
      --data_path "$DATA_PATH" --save_dir out --checkpoint_dir checkpoints/ablations \
      --from_weight sft_full_vlm --save_weight "$name" \
      --epochs 1 --group_size 4 --ppo_epochs 2 --learning_rate 1e-6 \
      --clip_eps 0.2 --max_prompt_length 384 --max_new_tokens 48 \
      --num_workers 4 --save_interval 250 --log_interval 10 \
      --resume "$resume" --seed "$SEED" \
      --log_file "logs/ablations/${name}.jsonl" "$@" \
      2>&1 | tee -a "logs/ablations/${name}.log"
    touch "logs/ablations/${name}.done"
  fi
}

run_eval() {
  local name="$1"
  if [[ ! -f "results/grpo_stability/coco500/${name}/summary.json" ]]; then
    "$PYTHON_BIN" -u eval_vlm_benchmark.py \
      --annotation_file "$COCO_ANNOTATIONS" --image_dir "$COCO_IMAGES" \
      --weight "$name" --save_dir out --max_samples 500 --max_new_tokens 48 \
      --conditions correct black shuffled --seed "$SEED" \
      --output_dir results/grpo_stability/coco500 \
      2>&1 | tee "logs/ablations/eval_${name}.log"
  fi
}

# Controlled variant 1: only beta changes from 0.02 to 0.10.
run_grpo grpo_beta010_vlm \
  --beta 0.10 --adaptive_beta 0 --kl_stop 0

# Controlled variant 2: beta is adjusted toward KL=0.10, with a safety stop.
run_grpo grpo_adaptive_vlm \
  --beta 0.02 --adaptive_beta 1 --target_kl 0.10 \
  --beta_update_rate 0.01 --min_beta 0.01 --max_beta 0.50 \
  --kl_stop 0.50 --kl_stop_patience 50

run_eval grpo_beta010_vlm
run_eval grpo_adaptive_vlm

touch logs/grpo_stability.done
echo "GRPO stability ablation complete."
