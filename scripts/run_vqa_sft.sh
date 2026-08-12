#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
SEED="${SEED:-42}"
NAME="vqa_sft_vlm"

for _ in $(seq 1 90); do
  test -f dataset/vqa_sft.parquet && break
  sleep 10
done
test -f dataset/vqa_sft.parquet || { echo "dataset/vqa_sft.parquet missing"; exit 1; }

mkdir -p logs/vqa checkpoints/vqa results/vqa out
resume=0
[[ -f "checkpoints/vqa/${NAME}_768_resume.pth" ]] && resume=1
if [[ ! -f "logs/vqa/${NAME}.done" ]]; then
  "$PYTHON_BIN" -u trainer/train_sft_vlm.py \
    --data_path dataset/vqa_sft.parquet --save_dir out --checkpoint_dir checkpoints/vqa \
    --from_weight sft_full_vlm --save_weight "$NAME" \
    --epochs 1 --batch_size 64 --learning_rate 5e-6 --freeze_llm 1 \
    --num_workers 8 --prefetch_factor 4 --persistent_workers 1 \
    --save_interval 1000 --log_interval 100 --from_resume "$resume" --seed "$SEED" \
    2>&1 | tee -a "logs/vqa/${NAME}.log"
  touch "logs/vqa/${NAME}.done"
fi

echo "== VQA eval =="
"$PYTHON_BIN" -u scripts/eval_vqa.py \
  --questions_file dataset/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
  --annotations_file dataset/vqav2/v2_mscoco_val2014_annotations.json \
  --image_zip /autodl-pub/data/COCO14/val2014.zip \
  --weight "$NAME" --save_dir out --projector_type mlp \
  --max_samples 2000 --max_new_tokens 32 --seed "$SEED" \
  --output_dir results/vqa \
  2>&1 | tee "logs/vqa/eval_vqa_${NAME}.log"

echo "== official COCO eval (caption retention) =="
"$PYTHON_BIN" -u scripts/eval_coco_official.py \
  --annotation_file dataset/coco2017/annotations/captions_val2017.json \
  --image_dir dataset/coco2017/val2017 \
  --weight "$NAME" --save_dir out --projector_type mlp \
  --max_new_tokens 48 --output_dir results/official_coco \
  2>&1 | tee "logs/vqa/eval_coco_${NAME}.log"

echo "VQA SFT pipeline complete."
