#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ANNOTATION="dataset/coco2017/annotations/captions_val2017.json"
IMAGE_DIR="dataset/coco2017/val2017"
VQA_QUESTIONS="dataset/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"
VQA_ANNOTATIONS="dataset/vqav2/v2_mscoco_val2014_annotations.json"
VQA_IMAGE_ZIP="/autodl-pub/data/COCO14/val2014.zip"

# Wait for the fix2 internal evaluation (launched by run_grpo_fix2.sh) to finish.
for _ in $(seq 1 90); do
  test -f results/grpo_fix2/coco500/grpo_fix2_vlm/summary.json && break
  sleep 10
done
echo "== fix2 internal eval ready =="

echo "== official COCO eval: grpo_fix2_vlm =="
"$PYTHON_BIN" -u scripts/eval_coco_official.py \
  --annotation_file "$ANNOTATION" --image_dir "$IMAGE_DIR" \
  --weight grpo_fix2_vlm --save_dir out --projector_type mlp \
  --max_new_tokens 48 --output_dir results/official_coco \
  2>&1 | tee logs/official_eval_grpo_fix2_vlm.log

for weight in sft_full_vlm grpo_full_vlm grpo_fix_vlm grpo_fix2_vlm; do
  echo "== VQA eval: $weight =="
  "$PYTHON_BIN" -u scripts/eval_vqa.py \
    --questions_file "$VQA_QUESTIONS" --annotations_file "$VQA_ANNOTATIONS" \
    --image_zip "$VQA_IMAGE_ZIP" \
    --weight "$weight" --save_dir out --projector_type mlp \
    --max_samples 2000 --max_new_tokens 32 --seed 42 \
    --output_dir results/vqa \
    2>&1 | tee "logs/vqa_eval_${weight}.log"
done

echo "VQA + official fix2 evaluation complete."
