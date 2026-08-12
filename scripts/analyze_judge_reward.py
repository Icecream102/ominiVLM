"""Analyze whether a stronger-model judge is a better caption reward proxy.

Samples predictions from an existing official-COCO run, scores each caption
1..5 with Qwen2.5-VL-3B, then correlates the judge score with official
COCOEvalCap metrics (CIDEr / ROUGE-L / BLEU-4) and with the internal n-gram
reference reward used by GRPO. High judge/metric correlation supports
replacing proxy rewards with a judge signal; low correlation explains the
earlier reward-misalignment finding.
"""

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.caption_metrics import meteor_exact, rouge_l, tokenize
from evaluation.grpo_rewards import reference_reward
from scipy.stats import spearmanr


def parse_args():
    parser = argparse.ArgumentParser(description="Judge reward proxy analysis")
    parser.add_argument("--predictions", required=True, help="predictions_coco.json from an official run")
    parser.add_argument("--annotation_file", default="dataset/coco2017/annotations/captions_val2017.json")
    parser.add_argument("--image_dir", default="dataset/coco2017/val2017")
    parser.add_argument("--model_path", default="model/qwen25vl-3b-instruct")
    parser.add_argument("--adapter_path", default="", help="optional LoRA adapter for the judge")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--output_dir", default="results/judge_analysis")
    parser.add_argument("--tag", default="default")
    return parser.parse_args()


def main():
    args = parse_args()
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    with open(args.predictions, encoding="utf-8") as file:
        predictions = {int(item["image_id"]): item["caption"] for item in json.load(file)}
    with open(args.annotation_file, encoding="utf-8") as file:
        raw = json.load(file)
    names = {item["id"]: item["file_name"] for item in raw["images"]}
    references = defaultdict(list)
    for item in raw["annotations"]:
        references[item["image_id"]].append(item["caption"])

    records = [
        {
            "image_id": image_id,
            "image_path": str(Path(args.image_dir) / names[image_id]),
            "caption": caption,
            "references": references[image_id],
        }
        for image_id, caption in predictions.items()
        if image_id in names and (Path(args.image_dir) / names[image_id]).is_file() and image_id in references
    ]
    random.Random(args.seed).shuffle(records)
    records = records[: args.samples]
    print(f"analyzing {len(records)} captions")

    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="bfloat16", device_map="cuda"
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    judge_scores = []
    start = time.perf_counter()
    for index, record in enumerate(records):
        image = Image.open(record["image_path"]).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "Rate this image description from 1 to 5 for factual accuracy, "
                    f"completeness and naturalness. Output only the integer score.\nDescription: {record['caption']}"
                )},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        response = processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        match = re.search(r"\d+(\.\d+)?", response)
        judge_scores.append(float(match.group(0)) if match else 3.0)
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f"{index + 1}/{len(records)} ({time.perf_counter() - start:.0f}s)")

    for index, record in enumerate(records):
        record["judge_score"] = judge_scores[index]

    print("computing per-caption metrics...")
    gts = {record["image_id"]: record["references"] for record in records}
    res = {record["image_id"]: [record["caption"]] for record in records}
    ordered_ids = list(gts.keys())

    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    _, per_bleu = Bleu(4).compute_score(gts, res)          # per_bleu[n-1][i]
    _, per_cider = Cider().compute_score(gts, res)          # list aligned with gts order
    per_bleu4 = {image_id: float(per_bleu[3][position]) for position, image_id in enumerate(ordered_ids)}
    per_cider_scores = {image_id: float(per_cider[position]) for position, image_id in enumerate(ordered_ids)}

    per_rouge = {}
    per_meteor = {}
    proxy = []
    for record in records:
        ref_tokens = [tokenize(ref) for ref in record["references"]]
        per_rouge[record["image_id"]] = rouge_l(tokenize(record["caption"]), ref_tokens)
        per_meteor[record["image_id"]] = meteor_exact(tokenize(record["caption"]), ref_tokens)
        base, _ = reference_reward(record["caption"], record["references"], weights={
            "unigram_f1": 0.10, "rouge_l": 0.30, "meteor_exact": 0.30,
            "length_score": 0.05, "repetition_penalty": 0.10,
        })
        proxy.append(base)

    metric_values = {
        "rouge_l": [per_rouge[record["image_id"]] for record in records],
        "meteor_exact": [per_meteor[record["image_id"]] for record in records],
        "bleu4": [per_bleu4[record["image_id"]] for record in records],
        "cider": [per_cider_scores[record["image_id"]] for record in records],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{args.tag}_records.jsonl", "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    correlations = {"judge_vs_proxy_reward": spearmanr(judge_scores, proxy).correlation}
    print(f"judge vs internal proxy reward: Spearman r = {correlations['judge_vs_proxy_reward']:.4f}")
    for metric, values in metric_values.items():
        r = spearmanr(judge_scores, values).correlation
        correlations[f"judge_vs_{metric}"] = r
        print(f"judge vs {metric}: Spearman r = {r:.4f}")

    summary = {
        "tag": args.tag,
        "samples": len(records),
        "judge_mean": sum(judge_scores) / max(len(judge_scores), 1),
        "judge_std": (sum((s - sum(judge_scores) / len(judge_scores)) ** 2 for s in judge_scores)
                      / max(len(judge_scores), 1)) ** 0.5,
        "per_caption_metric_means": {metric: sum(values) / len(values) for metric, values in metric_values.items()},
        "proxy_reward_mean": sum(proxy) / max(len(proxy), 1),
        "correlations": correlations,
        "note": "Spearman rank correlation between Qwen2.5-VL-3B judge scores and official metrics / proxy reward.",
    }
    with open(output_dir / f"{args.tag}_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
