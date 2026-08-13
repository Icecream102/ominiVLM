#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
SEED="${SEED:-42}"
SOURCE="dataset/pretrain_i2t.parquet"
SUBSET="dataset/pretrain_ablation_64k.parquet"
RESULT_ROOT="results/architecture_ablation/coco500"

mkdir -p logs/architecture checkpoints/architecture "$RESULT_ROOT" out

if [[ ! -f "$SUBSET" ]]; then
  "$PYTHON_BIN" scripts/build_grpo_dataset.py \
    --input "$SOURCE" --output "$SUBSET" --samples 64000 --seed "$SEED" \
    2>&1 | tee logs/architecture/build_subset.log
fi

run_pretrain() {
  local name="$1"
  local projector="$2"
  local vision_layers="$3"
  local batch_size="$4"
  local accumulation="$5"
  local max_steps="$6"
  local resume=0
  [[ -f "checkpoints/architecture/${name}_768_resume.pth" ]] && resume=1
  if [[ ! -f "logs/architecture/${name}.done" ]]; then
    "$PYTHON_BIN" -u trainer/train_pretrain_vlm.py \
      --data_path "$SUBSET" --save_dir out --checkpoint_dir checkpoints/architecture \
      --from_weight llm --save_weight "$name" --epochs 1 \
      --batch_size "$batch_size" --accumulation_steps "$accumulation" \
      --max_steps "$max_steps" --learning_rate 4e-4 --vision_learning_rate 1e-5 \
      --freeze_llm 2 --projector_type "$projector" \
      --vision_unfreeze_layers "$vision_layers" \
      --num_workers 8 --prefetch_factor 4 --persistent_workers 1 \
      --save_interval "$max_steps" --log_interval 50 \
      --from_resume "$resume" --seed "$SEED" \
      2>&1 | tee -a "logs/architecture/${name}.log"
    touch "logs/architecture/${name}.done"
  fi
}

run_eval() {
  local name="$1"
  local projector="$2"
  if [[ ! -f "$RESULT_ROOT/${name}/summary.json" ]]; then
    "$PYTHON_BIN" -u eval_vlm_benchmark.py \
      --annotation_file dataset/coco2017/annotations/captions_val2017.json \
      --image_dir dataset/coco2017/val2017 \
      --weight "$name" --save_dir out --projector_type "$projector" \
      --max_samples 500 --max_new_tokens 48 \
      --conditions correct black shuffled --seed "$SEED" \
      --output_dir "$RESULT_ROOT" \
      2>&1 | tee "logs/architecture/eval_${name}.log"
  fi
}

# All variants see exactly 64k samples and make 500 optimizer updates.
run_pretrain arch_linear_frozen linear 0 128 1 500
run_pretrain arch_mlp_frozen mlp 0 128 1 500
run_pretrain arch_mlp_vision2 mlp 2 32 4 2000

run_eval arch_linear_frozen linear
run_eval arch_mlp_frozen mlp
run_eval arch_mlp_vision2 mlp

touch logs/architecture_ablation.done
echo "Architecture ablation complete."
