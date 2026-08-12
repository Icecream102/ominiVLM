"""VQAv2 open-ended evaluation for Qwen2.5-VL + LoRA adapter."""

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
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="VQAv2 eval for Qwen2.5-VL LoRA")
    parser.add_argument("--model_path", default="model/qwen25vl-3b-instruct")
    parser.add_argument("--adapter_path", default="out/qwen_vl_lora")
    parser.add_argument("--questions_file", required=True)
    parser.add_argument("--annotations_file", required=True)
    parser.add_argument("--image_zip", required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/vqa_qwen")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    return parser.parse_args()


def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    args = parse_args()
    with open(args.questions_file, encoding="utf-8") as file:
        questions = json.load(file)["questions"]
    with open(args.annotations_file, encoding="utf-8") as file:
        annotations = json.load(file)["annotations"]
    answers_by_question = defaultdict(list)
    for item in annotations:
        answers_by_question[item["question_id"]] = [
            normalize_answer(answer["answer"]) for answer in item["answers"]
        ]
    records = [
        {"question_id": q["question_id"], "image_id": q["image_id"], "question": q["question"]}
        for q in questions if q["question_id"] in answers_by_question
    ]
    random.Random(args.seed).shuffle(records)
    records = records[: args.max_samples]

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
    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as output:
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
            answer = processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            normalized = normalize_answer(answer)
            matches = sum(1 for human in answers_by_question[record["question_id"]] if human == normalized)
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

    summary = {"weight": "qwen2.5-vl-3b-lora", "samples": len(records), "accuracy": correct / max(len(records), 1)}
    with open(output_dir / "summary.json", "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
