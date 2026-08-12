#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
SEED="${SEED:-42}"
DATA_PATH="${DATA_PATH:-dataset/grpo_i2t.parquet}"
NAME="grpo_fix2_vlm"

mkdir -p logs/fix checkpoints/fix results/grpo_fix2/coco500 out

resume=0
[[ -f "checkpoints/fix/${NAME}_768_resume.pth" ]] && resume=1
if [[ ! -f "logs/fix/${NAME}.done" ]]; then
  "$PYTHON_BIN" -u trainer/train_grpo_vlm.py \
    --data_path "$DATA_PATH" --save_dir out --checkpoint_dir checkpoints/fix \
    --from_weight sft_full_vlm --save_weight "$NAME" \
    --epochs 1 --group_size 16 --ppo_epochs 2 --learning_rate 1e-6 \
    --clip_eps 0.2 --beta 0.02 --adaptive_beta 1 --target_kl 0.06 \
    --beta_update_rate 0.02 --min_beta 0.01 --max_beta 0.60 \
    --kl_stop 0.50 --kl_stop_patience 50 \
    --reward_unigram 0.10 --reward_rouge 0.30 --reward_meteor 0.30 \
    --reward_cider 0.25 --reward_length 0.05 --reward_repetition 0.10 \
    --max_prompt_length 384 --max_new_tokens 48 \
    --num_workers 4 --save_interval 250 --log_interval 10 \
    --resume "$resume" --seed "$SEED" \
    --log_file "logs/fix/${NAME}.jsonl" \
    2>&1 | tee -a "logs/fix/${NAME}.log"
  touch "logs/fix/${NAME}.done"
fi

"$PYTHON_BIN" -u eval_vlm_benchmark.py \
  --annotation_file dataset/coco2017/annotations/captions_val2017.json \
  --image_dir dataset/coco2017/val2017 \
  --weight "$NAME" --save_dir out --max_samples 500 --max_new_tokens 48 \
  --conditions correct black shuffled --seed "$SEED" \
  --output_dir results/grpo_fix2/coco500 \
  2>&1 | tee "logs/fix/eval_${NAME}.log"

echo "GRPO fix2 run complete."
