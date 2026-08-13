"""TextVQA evaluation for Qwen2.5-VL + LoRA/DPO adapters.

TextVQA is a real-OCR benchmark: questions require reading text rendered in
the image. Images are COCO2014 val2014 (read in-place from the AutoDL zip,
no extra image download). Scoring follows the official VQA 3-of-10 rule.
"""

import argparse
import json
import random
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="TextVQA eval for Qwen2.5-VL")
    parser.add_argument("--model_path", default="model/qwen25vl-7b-instruct")
    parser.add_argument("--adapter_path", default="out/qwen7b_qlora_multitask")
    parser.add_argument("--annotations_file", default="dataset/textvqa/TextVQA_0.5.1_val.json")
    parser.add_argument("--image_zip", default="/autodl-pub/data/COCO14/val2014.zip")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all questions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output_dir", default="results/textvqa")
    parser.add_argument("--tag", default="qwen7b-qlora")
    return parser.parse_args()


def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    args = parse_args()
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    with open(args.annotations_file, encoding="utf-8") as file:
        data = json.load(file)
    records = []
    for item in data["data"]:
        image_file = item.get("image", "")
        match = re.search(r"(\d{12})\.jpg$", image_file)
        if not match:
            continue
        records.append({
            "question_id": item.get("question_id"),
            "image_id": int(match.group(1)),
            "question": item["question"],
            "answers": [normalize_answer(a) for a in item.get("answers", [])],
        })
    records = [record for record in records if record["answers"]]
    random.Random(args.seed).shuffle(records)
    if args.max_samples:
        records = records[: args.max_samples]
    print(f"evaluating {len(records)} TextVQA questions")

    processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="bfloat16", device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    archive = zipfile.ZipFile(args.image_zip)
    by_id = {}
    for name in archive.namelist():
        match = re.search(r"(\d{12})\.jpg$", name)
        if match:
            by_id[int(match.group(1))] = name

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    correct = 0
    start = time.perf_counter()
    with open(output_dir / f"{args.tag}_predictions.jsonl", "w", encoding="utf-8") as output:
        for index, record in enumerate(records):
            image = Image.open(archive.open(by_id[record["image_id"]])).convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": record["question"] + " Answer in one word or a short phrase."},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            answer = processor.decode(
                generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            normalized = normalize_answer(answer)
            matches = sum(1 for human in record["answers"] if human == normalized)
            score = min(1.0, matches / 3)
            correct += score
            output.write(json.dumps({
                "question_id": record["question_id"],
                "question": record["question"],
                "answer": answer,
                "normalized_answer": normalized,
                "matches": matches,
                "score": score,
            }, ensure_ascii=False) + "\n")
            if (index + 1) % 100 == 0 or index + 1 == len(records):
                elapsed = time.perf_counter() - start
                print(f"{index + 1}/{len(records)} acc={correct / (index + 1):.4f} ({elapsed:.0f}s)")

    summary = {
        "tag": args.tag,
        "samples": len(records),
        "accuracy": correct / max(len(records), 1),
        "note": "TextVQA 0.5.1 val; official 3-of-10 agreement rule with answer normalization.",
    }
    with open(output_dir / f"{args.tag}_summary.json", "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
