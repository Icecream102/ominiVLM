#!/usr/bin/env bash
# From-scratch VLM (option 3): random projector + FULLY UNFROZEN LLM alignment
# on 1.27M image-text pairs, then multi-task SFT, then full evaluation.
# Vision encoder stays frozen (SigLIP2) — standard practice.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p out checkpoints logs/scratch results/scratch

echo "== stage 1: from-scratch VLM pretrain (freeze_llm=0) =="
"$PYTHON_BIN" -u trainer/train_pretrain_vlm.py \
  --data_path dataset/pretrain_i2t.parquet \
  --save_dir out --checkpoint_dir checkpoints/scratch \
  --from_weight llm --save_weight scratch_vlm \
  --epochs 1 --batch_size 64 --accumulation_steps 2 --learning_rate 4e-4 \
  --freeze_llm 0 --projector_type mlp --max_seq_len 450 \
  --num_workers 4 --save_interval 5000 --log_interval 200 \
  --from_resume 0 --seed 42 \
  2>&1 | tee logs/scratch/pretrain_scratch.log

echo "== stage 2: build scratch SFT mix (multitask + hallucination + spatial + OCR) =="
EXTRA_SFT=""
for f in dataset/hallucination_sft.parquet dataset/spatial_qa.parquet dataset/synthetic_ocr.parquet; do
  [[ -f "$f" ]] && EXTRA_SFT="$EXTRA_SFT $f"
done
# shellcheck disable=SC2086
"$PYTHON_BIN" scripts/concat_parquets.py \
  --inputs dataset/multitask_sft.parquet $EXTRA_SFT \
  --output dataset/scratch_sft_mix.parquet

echo "== stage 3: multi-task SFT from scratch_vlm =="
"$PYTHON_BIN" -u trainer/train_sft_vlm.py \
  --data_path dataset/scratch_sft_mix.parquet \
  --save_dir out --checkpoint_dir checkpoints/scratch \
  --from_weight scratch_vlm --save_weight scratch_sft_vlm \
  --epochs 1 --batch_size 32 --accumulation_steps 2 \
  --learning_rate 5e-6 --freeze_llm 1 --projector_type mlp \
  --max_seq_len 768 --num_workers 4 \
  --save_interval 2000 --log_interval 200 \
  --from_resume 0 --seed 42 \
  2>&1 | tee logs/scratch/sft_scratch.log

echo "== stage 4: evaluation battery =="
WEIGHT=scratch_sft_vlm bash -c '
  PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
  "$PYTHON_BIN" -u scripts/eval_coco_official.py \
    --annotation_file dataset/coco2017/annotations/captions_val2017.json \
    --image_dir dataset/coco2017/val2017 --weight scratch_sft_vlm \
    --save_dir out --projector_type mlp --max_new_tokens 48 \
    --output_dir results/official_coco --with_meteor --meteor_timeout 900 \
    2>&1 | tee logs/scratch/official_coco_scratch.log
  "$PYTHON_BIN" -u scripts/eval_pope.py --model minimind \
    --instances_file dataset/coco2017/annotations/instances_val2017.json \
    --image_dir dataset/coco2017/val2017 --save_dir out \
    --weight scratch_sft_vlm --output_dir results/pope --tag scratch_sft_vlm \
    2>&1 | tee logs/scratch/pope_scratch.log
  "$PYTHON_BIN" -u scripts/eval_text_only.py --weight scratch_sft_vlm --max_samples 500 \
    2>&1 | tee logs/scratch/text_only_scratch.log
  "$PYTHON_BIN" -u scripts/eval_mmbench.py --model minimind \
    --dataset_dir dataset/mmbench_en_dev --save_dir out --weight scratch_sft_vlm \
    --output_dir results/mmbench --tag scratch_sft_vlm \
    2>&1 | tee logs/scratch/mmbench_scratch.log
'

echo "from-scratch VLM pipeline complete."
