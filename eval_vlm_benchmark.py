"""Evaluate native MiniMind-V checkpoints on COCO-style caption data.

In one run the script evaluates correct images and optional visual controls
(black images and deterministically shuffled images). This distinguishes text
generation quality from actual visual dependence.
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from evaluation.caption_metrics import compute_caption_metrics, jaccard_distance
from model.torch_compat import ensure_torch_transformers_compat
ensure_torch_transformers_compat()
from transformers import AutoTokenizer

from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.trainer_utils import get_model_params, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind-V COCO caption benchmark")
    parser.add_argument("--annotation_file", required=True, help="COCO captions JSON")
    parser.add_argument("--image_dir", required=True, help="COCO image directory")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="sft_vlm")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--vision_model_path", default="model/siglip2-base-p32-256-ve")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--projector_type", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--prompt", default="<image>\nDescribe this image in one concise sentence.")
    parser.add_argument("--conditions", nargs="+", choices=["correct", "black", "shuffled"], default=["correct", "black", "shuffled"])
    parser.add_argument("--output_dir", default="benchmark_results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_records(annotation_file, image_dir, max_samples, seed):
    with open(annotation_file, encoding="utf-8") as file:
        raw = json.load(file)
    names = {item["id"]: item["file_name"] for item in raw["images"]}
    captions = defaultdict(list)
    for item in raw["annotations"]:
        captions[item["image_id"]].append(item["caption"])
    records = [
        {
            "image_id": image_id,
            "image_path": str(Path(image_dir) / names[image_id]),
            "references": refs,
        }
        for image_id, refs in captions.items()
        if image_id in names and (Path(image_dir) / names[image_id]).is_file()
    ]
    random.Random(seed).shuffle(records)
    return records[:max_samples] if max_samples else records


def load_model(args):
    suffix = "_moe" if args.use_moe else ""
    checkpoint = Path(args.save_dir) / f"{args.weight}_{args.hidden_size}{suffix}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
        projector_type=args.projector_type,
    )
    model = MiniMindVLM(config, vision_model_path=args.vision_model_path)
    if model.vision_encoder is None:
        raise FileNotFoundError(f"vision model not found: {args.vision_model_path}")
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict({key: value for key, value in state_dict.items() if "mask" not in key}, strict=False)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model = model.eval().to(args.device)
    if args.device.startswith("cuda"):
        model = model.to(dtype)
    get_model_params(model, config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    return model, tokenizer, model.processor


def condition_image(records, index, condition):
    source_index = (index + 1) % len(records) if condition == "shuffled" else index
    image = Image.open(records[source_index]["image_path"]).convert("RGB")
    if condition == "black":
        image = Image.new("RGB", image.size, color=(0, 0, 0))
    return image, records[source_index]["image_id"]


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def run_condition(args, model, tokenizer, processor, records, condition, output_path):
    marker = model.config.image_special_token * model.config.image_token_len
    messages = [{"role": "user", "content": args.prompt.replace("<image>", marker)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(args.device)
    predictions, references, latencies = [], [], []
    generated_tokens = 0
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    with open(output_path, "w", encoding="utf-8") as output:
        for index, record in enumerate(records):
            image, source_image_id = condition_image(records, index, condition)
            preprocess_start = time.perf_counter()
            pixel_values = {
                key: value.to(args.device)
                for key, value in MiniMindVLM.image2tensor(image, processor).items()
            }
            synchronize(args.device)
            generation_start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=pixel_values,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            synchronize(args.device)
            generation_seconds = time.perf_counter() - generation_start
            end_to_end_seconds = time.perf_counter() - preprocess_start
            new_ids = generated[0, inputs["input_ids"].shape[1]:]
            prediction = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            predictions.append(prediction)
            references.append(record["references"])
            generated_tokens += len(new_ids)
            latencies.append(end_to_end_seconds)
            output.write(json.dumps({
                "index": index,
                "target_image_id": record["image_id"],
                "source_image_id": source_image_id,
                "condition": condition,
                "prediction": prediction,
                "references": record["references"],
                "generation_seconds": generation_seconds,
                "end_to_end_seconds": end_to_end_seconds,
                "generated_tokens": len(new_ids),
            }, ensure_ascii=False) + "\n")
            if (index + 1) % 25 == 0 or index + 1 == len(records):
                print(f"[{condition}] {index + 1}/{len(records)}")

    scores = compute_caption_metrics(predictions, references)
    total_seconds = sum(latencies)
    scores.update({
        "samples": len(records),
        "avg_end_to_end_seconds": total_seconds / max(len(records), 1),
        "tokens_per_second": generated_tokens / max(total_seconds, 1e-9),
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / 1024 ** 2
            if args.device.startswith("cuda") else 0.0
        ),
    })
    return scores, predictions


def main():
    args = parse_args()
    setup_seed(args.seed)
    records = load_records(args.annotation_file, args.image_dir, args.max_samples, args.seed)
    if len(records) < 2:
        raise RuntimeError("at least two valid COCO records are required")
    output_dir = Path(args.output_dir) / args.weight
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, processor = load_model(args)

    condition_metrics, condition_predictions = {}, {}
    for condition in args.conditions:
        metrics, predictions = run_condition(
            args, model, tokenizer, processor, records, condition,
            output_dir / f"predictions_{condition}.jsonl",
        )
        condition_metrics[condition] = metrics
        condition_predictions[condition] = predictions

    visual_dependency = {}
    correct = condition_predictions.get("correct")
    if correct:
        for condition in ("black", "shuffled"):
            altered = condition_predictions.get(condition)
            if not altered:
                continue
            visual_dependency[condition] = {
                "output_change_rate": sum(a != b for a, b in zip(correct, altered)) / len(correct),
                "mean_token_jaccard_distance": sum(
                    jaccard_distance(a, b) for a, b in zip(correct, altered)
                ) / len(correct),
                "CIDEr_drop": condition_metrics["correct"]["CIDEr"] - condition_metrics[condition]["CIDEr"],
                "BLEU-4_drop": condition_metrics["correct"]["BLEU-4"] - condition_metrics[condition]["BLEU-4"],
            }

    summary = {
        "weight": args.weight,
        "checkpoint": str(Path(args.save_dir) / f"{args.weight}_{args.hidden_size}{'_moe' if args.use_moe else ''}.pth"),
        "annotation_file": args.annotation_file,
        "prompt": args.prompt,
        "seed": args.seed,
        "conditions": condition_metrics,
        "visual_dependency": visual_dependency,
        "metric_note": "Internal exact-token metrics; use official COCOEvalCap for paper comparison.",
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
