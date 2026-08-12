#!/usr/bin/env bash
# Final multi-task SFT for the 65M MiniMind-V: caption + VQAv2 + OK-VQA + MMBench.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NAME="multitask_final_vlm"
mkdir -p out checkpoints logs

resume=0
[[ -f "checkpoints/${NAME}_768_resume.pth" ]] && resume=1

"$PYTHON_BIN" -u trainer/train_sft_vlm.py \
  --data_path dataset/multitask_sft.parquet \
  --save_dir out --checkpoint_dir checkpoints \
  --from_weight pretrain_full_vlm --save_weight "$NAME" \
  --epochs 2 --batch_size 32 --accumulation_steps 2 \
  --learning_rate 5e-6 --freeze_llm 1 --projector_type mlp \
  --max_seq_len 768 --num_workers 8 \
  --save_interval 2000 --log_interval 200 \
  --from_resume "$resume" --seed 42 \
  2>&1 | tee "logs/${NAME}.log"

echo "multitask SFT complete."
