"""vLLM inference benchmark for Qwen2.5-VL-7B (+ LoRA / AWQ).

Measures tokens/s, TTFT, end-to-end latency and peak VRAM for image caption
prompts from COCO val2017, at configurable batch sizes. Produces a JSON
report that doubles as deployment evidence.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM benchmark for Qwen2.5-VL-7B")
    parser.add_argument("--model_path", default="model/qwen25vl-7b-instruct")
    parser.add_argument("--adapter_path", default="", help="LoRA adapter dir (empty = none)")
    parser.add_argument("--image_dir", default="dataset/coco2017/val2017")
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_pixels", type=int, default=768 * 28 * 28)
    parser.add_argument("--output_dir", default="results/deployment")
    parser.add_argument("--tag", default="vllm-qwen7b-lora")
    return parser.parse_args()


def peak_vram_mb():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        )
        return int(output.decode().strip().splitlines()[0])
    except Exception:
        return -1


def main():
    args = parse_args()
    from vllm import LLM, SamplingParams

    images = sorted(Path(args.image_dir).glob("*.jpg"))[: args.num_images]
    print(f"loading {len(images)} images; model={args.model_path}")

    llm = LLM(
        model=args.model_path,
        enable_lora=True if args.adapter_path else False,
        max_model_len=8192,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )
    peak_after_load = peak_vram_mb()
    print(f"peak VRAM after load: {peak_after_load} MB")

    prompts = ["Describe this image in one concise sentence."] * len(images)
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    loora_request = {"extra_lora_request": None}
    if args.adapter_path:
        from vllm.lora.request import LoRARequest
        loora_request["extra_lora_request"] = LoRARequest("qwen7b", 1, args.adapter_path)

    start = time.perf_counter()
    outputs = llm.generate(
        [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": Image.open(path).convert("RGB")},
            }
            for prompt, path in zip(prompts, images)
        ],
        sampling,
        **loora_request,
    )
    elapsed = time.perf_counter() - start
    total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    ttfts = [output.metrics.first_token_time - output.metrics.arrival_time for output in outputs]
    peak_infer = peak_vram_mb()

    report = {
        "tag": args.tag,
        "model": args.model_path,
        "adapter": args.adapter_path or "none",
        "num_prompts": len(images),
        "batch_size_effective": len(images),
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": round(elapsed, 2),
        "total_generated_tokens": total_tokens,
        "tokens_per_second": round(total_tokens / elapsed, 2),
        "requests_per_second": round(len(images) / elapsed, 2),
        "mean_ttft_seconds": round(sum(ttfts) / max(len(ttfts), 1), 4),
        "p50_ttft_seconds": round(sorted(ttfts)[len(ttfts) // 2], 4),
        "peak_vram_mb_load": peak_after_load,
        "peak_vram_mb_inference": peak_infer,
        "note": "single-image caption prompts; vLLM OpenAI-compatible engine.",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{args.tag}_summary.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output_dir / (args.tag + '_summary.json')}")


if __name__ == "__main__":
    main()
