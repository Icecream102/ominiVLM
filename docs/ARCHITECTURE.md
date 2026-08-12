# Architecture and experiment lifecycle

## Model path

```text
256x256 image
    -> frozen SigLIP2 vision encoder (64 visual tokens)
    -> LayerNorm + MLP Projection
    -> LLM embedding space
    -> 8-layer, 768-hidden MiniMind language model
    -> autoregressive multimodal response
```

The dense model has roughly 65M language-and-projection parameters. The vision encoder remains frozen in all three stages so the experiment isolates cross-modal alignment and language-policy changes.

## Training path

```text
LLM checkpoint
    -> Pretrain: image-caption alignment, Projection only
    -> SFT: multimodal instruction following, Projection + first/last LLM layers
    -> GRPO: grouped online rollouts, clipped objective + reference-policy KL
    -> COCO500: correct / black / shuffled controlled evaluation
```

Every stage writes an inference checkpoint and an atomic resume checkpoint. Done markers enforce the stage order and make `scripts/run_full_training_pipeline.sh` idempotent after interruption.

## Evaluation design

- Quality: BLEU-1/2/3/4, METEOR-exact, ROUGE-L and internal CIDEr-style.
- Efficiency: end-to-end latency, generated tokens/s and peak allocated VRAM.
- Visual dependence: replace the image with a black image or deterministic mismatch, then measure quality drop, output-change rate and token Jaccard distance.
- GRPO stability: reward, KL, clip fraction, completion length, beta and reward components are logged per step.

The repository metrics are dependency-light and intended for controlled checkpoint comparison. Official benchmark submissions should use the evaluator and tokenizer mandated by that benchmark.
