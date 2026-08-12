#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
SEED="${SEED:-42}"
mkdir -p checkpoints/vqa logs/vqa results/vqa_ratio results/vqa_ratio_coco500 out

run_variant() {
  local name="$1" vqa_n="$2" base_n="$3"
  local parquet="dataset/vqa_mix_${name}.parquet"
  if [[ ! -f "$parquet" ]]; then
    "$PYTHON_BIN" scripts/build_mixed_sft_dataset.py \
      --vqa_parquet dataset/vqa_sft.parquet --base_parquet dataset/sft_i2t.parquet \
      --vqa_samples "$vqa_n" --base_samples "$base_n" --seed "$SEED" --output "$parquet"
  fi
  local resume=0
  [[ -f "checkpoints/vqa/${name}_768_resume.pth" ]] && resume=1
  if [[ ! -f "logs/vqa/${name}.done" ]]; then
    "$PYTHON_BIN" -u trainer/train_sft_vlm.py \
      --data_path "$parquet" --save_dir out --checkpoint_dir checkpoints/vqa \
      --from_weight sft_full_vlm --save_weight "$name" \
      --epochs 1 --batch_size 64 --learning_rate 5e-6 --freeze_llm 1 \
      --num_workers 8 --prefetch_factor 4 --persistent_workers 1 \
      --save_interval 1000 --log_interval 100 --from_resume "$resume" --seed "$SEED" \
      2>&1 | tee -a "logs/vqa/${name}.log"
    touch "logs/vqa/${name}.done"
  fi
  echo "== VQA eval $name =="
  "$PYTHON_BIN" -u scripts/eval_vqa.py \
    --questions_file dataset/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \
    --annotations_file dataset/vqav2/v2_mscoco_val2014_annotations.json \
    --image_zip /autodl-pub/data/COCO14/val2014.zip \
    --weight "$name" --save_dir out --projector_type mlp \
    --max_samples 1000 --max_new_tokens 32 --seed "$SEED" \
    --output_dir results/vqa_ratio \
    2>&1 | tee "logs/vqa/eval_vqa_${name}.log"
  echo "== COCO500 eval $name =="
  "$PYTHON_BIN" -u eval_vlm_benchmark.py \
    --annotation_file dataset/coco2017/annotations/captions_val2017.json \
    --image_dir dataset/coco2017/val2017 \
    --weight "$name" --save_dir out --projector_type mlp \
    --max_samples 500 --max_new_tokens 48 --conditions correct black shuffled --seed "$SEED" \
    --output_dir results/vqa_ratio_coco500 \
    2>&1 | tee "logs/vqa/eval_coco_${name}.log"
}

# VQA:caption mixing ratio ablation (total ~40k rows per variant).
run_variant vqa_mix_1to3 10000 30000
run_variant vqa_mix_3to1 30000 10000

echo "mix ratio ablation complete."
