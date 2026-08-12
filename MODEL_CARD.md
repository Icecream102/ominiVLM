# MiniMind-V full-pipeline model card

## Model

- Vision: frozen SigLIP2, 256×256 input, 64 visual tokens.
- Connector: LayerNorm + MLP Projection.
- Language backbone: MiniMind, hidden size 768, 8 layers, about 65M parameters excluding the frozen vision encoder.
- Training: Pretrain → SFT → GRPO on one RTX 5090.

## Intended use

Research, education, small-scale multimodal training experiments, controlled checkpoint comparison and inference prototyping. The model is not intended for safety-critical decisions or production deployment without additional evaluation.

## Evaluation

See [`benchmark_results/full_pipeline_20260811/REPORT.md`](benchmark_results/full_pipeline_20260811/REPORT.md). The formal COCO500 experiment found that the current GRPO proxy reward is misaligned with held-out caption quality. The GRPO checkpoint is therefore an experimental artifact, not the recommended default checkpoint.

## Recommended checkpoint

Use the Pretrain or SFT checkpoint according to task style and validate on a task-specific held-out set. Do not select the GRPO checkpoint solely from its training reward.

## Limitations

- Small language backbone and fixed 256×256 visual resolution.
- Frozen vision encoder and single-image input.
- Training data may contain synthetic biases and unknown coverage gaps.
- Internal Caption metrics are not official COCOEvalCap scores.
- No comprehensive safety, OCR, hallucination, multilingual or adversarial benchmark has been completed.
