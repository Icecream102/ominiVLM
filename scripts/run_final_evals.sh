#!/usr/bin/env bash
# Full benchmark battery for a MiniMind-V checkpoint (65M) plus 3B LoRA:
# official COCOEvalCap (with METEOR), VQAv2, OK-VQA, MMBench (full), POPE.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WEIGHT="${WEIGHT:-multitask_final_vlm}"
ANNOTATION="dataset/coco2017/annotations/captions_val2017.json"
IMAGE_DIR="dataset/coco2017/val2017"
VQA_QUESTIONS="dataset/vqav2/v2_OpenEnded_mscoco_val2014_questions.json"
VQA_ANNOTATIONS="dataset/vqav2/v2_mscoco_val2014_annotations.json"
OKVQA_QUESTIONS="dataset/okvqa/OpenEnded_mscoco_val2014_questions.json"
OKVQA_ANNOTATIONS="dataset/okvqa/mscoco_val2014_annotations.json"
VQA_IMAGE_ZIP="/autodl-pub/data/COCO14/val2014.zip"
mkdir -p logs/final

echo "== official COCOEvalCap (with METEOR): $WEIGHT =="
"$PYTHON_BIN" -u scripts/eval_coco_official.py \
  --annotation_file "$ANNOTATION" --image_dir "$IMAGE_DIR" \
  --weight "$WEIGHT" --save_dir out --projector_type mlp \
  --max_new_tokens 48 --output_dir results/official_coco \
  --with_meteor --meteor_timeout 900 \
  2>&1 | tee "logs/final/official_coco_${WEIGHT}.log"

echo "== VQAv2: $WEIGHT =="
"$PYTHON_BIN" -u scripts/eval_vqa.py \
  --questions_file "$VQA_QUESTIONS" --annotations_file "$VQA_ANNOTATIONS" \
  --image_zip "$VQA_IMAGE_ZIP" --weight "$WEIGHT" --save_dir out \
  --projector_type mlp --max_samples 2000 --max_new_tokens 32 --seed 42 \
  --output_dir results/vqa \
  2>&1 | tee "logs/final/vqa_${WEIGHT}.log"

echo "== OK-VQA: $WEIGHT =="
"$PYTHON_BIN" -u scripts/eval_okvqa.py --model minimind \
  --questions_file "$OKVQA_QUESTIONS" --annotations_file "$OKVQA_ANNOTATIONS" \
  --image_zip "$VQA_IMAGE_ZIP" --weight "$WEIGHT" --save_dir out \
  --projector_type mlp --output_dir results/okvqa \
  2>&1 | tee "logs/final/okvqa_${WEIGHT}.log"

echo "== MMBench en/dev full: $WEIGHT =="
"$PYTHON_BIN" -u scripts/eval_mmbench.py --model minimind \
  --dataset_dir dataset/mmbench_en_dev --save_dir out --weight "$WEIGHT" \
  --output_dir results/mmbench --tag "${WEIGHT}_full" \
  2>&1 | tee "logs/final/mmbench_${WEIGHT}.log"

echo "== POPE: $WEIGHT =="
"$PYTHON_BIN" -u scripts/eval_pope.py --model minimind \
  --instances_file dataset/coco2017/annotations/instances_val2017.json \
  --image_dir dataset/coco2017/val2017 \
  --save_dir out --weight "$WEIGHT" --output_dir results/pope \
  --tag "${WEIGHT}" \
  2>&1 | tee "logs/final/pope_${WEIGHT}.log"

echo "final eval battery complete."
