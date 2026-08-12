#!/usr/bin/env bash
# GRPO v3: replace n-gram/CIDEr proxy rewards with a stronger-model judge
# (Qwen2.5-VL-3B base). Same data and base policy as grpo_fix2, so the only
# controlled difference is the reward signal. Bounded run for single-GPU time.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NAME="grpo_judge_vlm"
mkdir -p logs/judge checkpoints/judge out

resume=0
[[ -f "checkpoints/judge/${NAME}_768_resume.pth" ]] && resume=1

"$PYTHON_BIN" -u trainer/train_grpo_vlm.py \
  --data_path dataset/grpo_i2t.parquet \
  --save_dir out --checkpoint_dir checkpoints/judge \
  --from_weight sft_full_vlm --save_weight "$NAME" \
  --epochs 1 --max_steps 400 --batch_size 1 --group_size 8 \
  --ppo_epochs 2 --learning_rate 1e-6 --clip_eps 0.2 \
  --beta 0.02 --adaptive_beta 1 --target_kl 0.08 \
  --beta_update_rate 0.05 --min_beta 0.01 --max_beta 0.60 \
  --kl_stop 0.50 --kl_stop_patience 50 \
  --reward_unigram 0.0 --reward_rouge 0.0 --reward_meteor 0.0 \
  --reward_length 0.0 --reward_repetition 0.0 --reward_cider 0.0 \
  --reward_judge 1.0 --judge_model_path model/qwen25vl-3b-instruct \
  --judge_max_new_tokens 4 \
  --max_prompt_length 384 --max_new_tokens 48 \
  --temperature 0.9 --top_p 0.95 --top_k 50 --repetition_penalty 1.05 \
  --num_workers 4 --save_interval 100 --log_interval 10 \
  --resume "$resume" --seed 42 \
  --log_file "logs/judge/${NAME}.jsonl" \
  2>&1 | tee "logs/judge/${NAME}.log"

echo "judge GRPO complete."
