#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ANNOTATION="dataset/coco2017/annotations/captions_val2017.json"
IMAGE_DIR="dataset/coco2017/val2017"
OUTPUT_DIR="results/official_coco"
mkdir -p "$OUTPUT_DIR"

# weight projector
WEIGHTS=(
  "pretrain_full_vlm mlp"
  "sft_full_vlm mlp"
  "grpo_full_vlm mlp"
  "grpo_adaptive_vlm mlp"
  "grpo_beta010_vlm mlp"
  "grpo_fix_vlm mlp"
)

for entry in "${WEIGHTS[@]}"; do
  read -r weight projector <<< "$entry"
  summary="$OUTPUT_DIR/$weight/summary.json"
  if [[ -f "$summary" ]]; then
    echo "skip $weight (already evaluated)"
    continue
  fi
  if [[ ! -f "out/${weight}_768.pth" ]]; then
    echo "skip $weight (checkpoint missing)"
    continue
  fi
  echo "== evaluating $weight ($projector) =="
  "$PYTHON_BIN" -u scripts/eval_coco_official.py \
    --annotation_file "$ANNOTATION" --image_dir "$IMAGE_DIR" \
    --weight "$weight" --save_dir out --projector_type "$projector" \
    --max_new_tokens 48 --output_dir "$OUTPUT_DIR" \
    2>&1 | tee "logs/official_eval_${weight}.log"
done

echo "official COCO evaluation complete."
