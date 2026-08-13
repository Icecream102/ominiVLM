"""Prove that DPO actually absorbed the preference signal.

Measures on the preference pairs (optionally a sampled subset):
  1. implicit reward margin: log_pi(chosen) - log_pi(rejected) for the
     QLoRA (pre-DPO) and DPO policies, plus the per-pair margin gain and
     the win rate (fraction of pairs where margin > 0);
  2. judge win rate: sample responses from both policies on the same
     prompts, score them with the Qwen3B judge, report how often the DPO
     model's response is judged better.
"""

import argparse
import io
import json
import random
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze DPO effect")
    parser.add_argument("--model_path", default="model/qwen25vl-7b-instruct")
    parser.add_argument("--base_adapter", default="out/qwen7b_qlora_multitask")
    parser.add_argument("--dpo_adapter", default="out/qwen7b_dpo")
    parser.add_argument("--data_path", default="dataset/preference_pairs.parquet")
    parser.add_argument("--max_pairs", type=int, default=400, help="subset for log-prob analysis")
    parser.add_argument("--judge_path", default="model/qwen25vl-3b-instruct")
    parser.add_argument("--judge_adapter", default="out/qwen_vl_lora")
    parser.add_argument("--gen_samples", type=int, default=200, help="prompts for judge win-rate check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/dpo_analysis")
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    return parser.parse_args()


def load_pairs(parquet_path, limit=0):
    table = pq.read_table(parquet_path)
    pairs = []
    for index in range(table.num_rows):
        image_bytes = table.column("image_bytes")[index].as_py()
        if isinstance(image_bytes, list):
            image_bytes = image_bytes[0]
        pairs.append({
            "image_bytes": image_bytes,
            "prompt": table.column("prompt")[index].as_py(),
            "chosen": table.column("chosen")[index].as_py(),
            "rejected": table.column("rejected")[index].as_py(),
        })
    return pairs[:limit] if limit else pairs


def build_inputs(processor, image, prompt, response):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": response},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = processor(text=text, images=image, return_tensors="pt")
    pixel_values = encoded["pixel_values"]
    if isinstance(pixel_values, (list, tuple)):
        pixel_values = torch.cat(
            [p if p.dim() == 4 else p.squeeze(0) for p in pixel_values], dim=0
        )
    elif pixel_values.dim() == 5:
        pixel_values = pixel_values[0]
    grid_thw = encoded["image_grid_thw"]
    if isinstance(grid_thw, (list, tuple)):
        grid_thw = torch.stack(grid_thw)
    if grid_thw.dim() == 1:
        grid_thw = grid_thw.unsqueeze(0)
    return {
        "input_ids": encoded["input_ids"].to("cuda"),
        "attention_mask": encoded["attention_mask"].to("cuda"),
        "pixel_values": pixel_values.to("cuda"),
        "image_grid_thw": grid_thw.to("cuda"),
    }


def sequence_log_prob(model, inputs):
    with torch.inference_mode():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
        )
    logits = outputs.logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    labels = inputs["input_ids"]
    gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100).float() if (labels != -100).any() else torch.ones_like(labels, dtype=torch.float)
    return (gathered * mask).sum() / mask.sum().clamp_min(1)


def main():
    args = parse_args()
    from peft import PeftModel
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    pairs = load_pairs(args.data_path)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    sample_pairs = pairs[: args.max_pairs]
    print(f"loaded {len(pairs)} pairs, analyzing {len(sample_pairs)}", flush=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)

    def load_policy(adapter_path):
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, quantization_config=bnb_config,
            torch_dtype=torch.bfloat16, device_map="cuda",
        )
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        return model

    print("loading base policy...", flush=True)
    base = load_policy(args.base_adapter)
    print("loading DPO policy...", flush=True)
    dpo = load_policy(args.dpo_adapter)

    margins = {"base": [], "dpo": []}
    start = time.perf_counter()
    for index, pair in enumerate(sample_pairs):
        image = Image.open(io.BytesIO(pair["image_bytes"])).convert("RGB")
        chosen_inputs = build_inputs(processor, image, pair["prompt"], pair["chosen"])
        rejected_inputs = build_inputs(processor, image, pair["prompt"], pair["rejected"])
        base_margin = sequence_log_prob(base, chosen_inputs) - sequence_log_prob(base, rejected_inputs)
        dpo_margin = sequence_log_prob(dpo, chosen_inputs) - sequence_log_prob(dpo, rejected_inputs)
        margins["base"].append(float(base_margin))
        margins["dpo"].append(float(dpo_margin))
        if (index + 1) % 50 == 0 or index + 1 == len(sample_pairs):
            print(f"{index + 1}/{len(sample_pairs)} ({time.perf_counter() - start:.0f}s)", flush=True)

    def stats(values):
        mean = sum(values) / len(values)
        win = sum(1 for value in values if value > 0) / len(values)
        return {"mean_margin": round(mean, 4), "win_rate": round(win, 4)}

    base_stats = stats(margins["base"])
    dpo_stats = stats(margins["dpo"])
    gains = [d - b for b, d in zip(margins["base"], margins["dpo"])]
    gain_stats = {
        "mean_margin_gain": round(sum(gains) / len(gains), 4),
        "improved_pair_ratio": round(sum(1 for g in gains if g > 0) / len(gains), 4),
    }
    print("base margins:", base_stats, flush=True)
    print("dpo margins:", dpo_stats, flush=True)
    print("gains:", gain_stats, flush=True)

    # Judge win-rate: generate from both policies on gen_samples prompts.
    print("loading judge...", flush=True)
    judge = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.judge_path, torch_dtype="bfloat16", device_map="cuda"
    )
    if args.judge_adapter:
        judge = PeftModel.from_pretrained(judge, args.judge_adapter)
    judge.eval()

    def generate(model, image, prompt):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=48, do_sample=True,
                temperature=0.8, top_p=0.95,
            )
        return processor.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

    def judge_score(image, prompt, response):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "Rate this image description from 1 to 5 for factual accuracy, "
                    f"completeness and naturalness. Output only the integer score.\nDescription: {response}"
                )},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = judge.generate(**inputs, max_new_tokens=4, do_sample=False)
        output = processor.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        match = re.search(r"\d+(\.\d+)?", output)
        return float(match.group(0)) if match else 3.0

    gen_pairs = pairs[: args.gen_samples]
    base_wins = dpo_wins = ties = 0
    judge_scores = {"base": [], "dpo": []}
    for index, pair in enumerate(gen_pairs):
        image = Image.open(io.BytesIO(pair["image_bytes"])).convert("RGB")
        base_response = generate(base, image, pair["prompt"])
        dpo_response = generate(dpo, image, pair["prompt"])
        score_base = judge_score(image, pair["prompt"], base_response)
        score_dpo = judge_score(image, pair["prompt"], dpo_response)
        judge_scores["base"].append(score_base)
        judge_scores["dpo"].append(score_dpo)
        if score_dpo > score_base:
            dpo_wins += 1
        elif score_dpo < score_base:
            base_wins += 1
        else:
            ties += 1
        if (index + 1) % 50 == 0 or index + 1 == len(gen_pairs):
            print(f"judge {index + 1}/{len(gen_pairs)} ({time.perf_counter() - start:.0f}s)", flush=True)

    judge_stats = {
        "samples": len(gen_pairs),
        "base_win": base_wins,
        "dpo_win": dpo_wins,
        "tie": ties,
        "dpo_win_rate": round(dpo_wins / max(len(gen_pairs), 1), 4),
        "base_mean_judge": round(sum(judge_scores["base"]) / max(len(judge_scores["base"]), 1), 4),
        "dpo_mean_judge": round(sum(judge_scores["dpo"]) / max(len(judge_scores["dpo"]), 1), 4),
    }
    print("judge stats:", judge_stats, flush=True)

    summary = {
        "pairs_analyzed": len(sample_pairs),
        "implicit_reward_margin": {"base": base_stats, "dpo": dpo_stats, "gain": gain_stats},
        "judge_win_rate": judge_stats,
        "note": "implicit reward margin = log_pi(chosen) - log_pi(rejected); "
                "judge win-rate compares sampled responses on held prompts.",
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dpo_effect_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {output_dir / 'dpo_effect_summary.json'}")


if __name__ == "__main__":
    main()
