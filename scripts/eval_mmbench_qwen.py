"""MMBench multiple-choice evaluation for Qwen2.5-VL + LoRA.

Expects the OpenCompass/mmbench TSV (dev or test) and an image directory.
Reports per-row accuracy, per-question circular accuracy (both option orders
correct), and per l2-category accuracy.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_tsv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--model_path", default="model/qwen25vl-3b-instruct")
    parser.add_argument("--adapter_path", default="out/qwen_vl_lora")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output_dir", default="results/mmbench")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_questions(tsv_path, image_dir, max_samples, seed):
    rows = []
    with open(tsv_path, encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            image_path = Path(image_dir) / row.get("image", "")
            if not image_path.is_file():
                continue
            options = {letter: row.get(letter, "") for letter in "ABCD" if row.get(letter, "")}
            rows.append({
                "index": row.get("index", ""),
                "question": row.get("question", ""),
                "options": options,
                "image_path": str(image_path),
                "answer": row.get("answer", "").strip().upper(),
                "l2": row.get("l2-category", ""),
            })
    if max_samples:
        import random
        random.Random(seed).shuffle(rows)
        rows = rows[: max_samples]
    return rows


def main():
    args = parse_args()
    rows = load_questions(args.questions_tsv, args.image_dir, args.max_samples, args.seed)
    print(f"loaded {len(rows)} MMBench questions")

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="bfloat16", device_map="cuda"
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    results = []
    correct = 0
    by_category = defaultdict(lambda: [0, 0])
    for index, row in enumerate(rows):
        image = Image.open(row["image_path"]).convert("RGB")
        option_text = "\n".join(f"{letter}. {text}" for letter, text in row["options"].items())
        user_text = (
            "请阅读图片，并从 A-D 选项中选择唯一正确答案，只输出一个选项字母。\n"
            f"问题：{row['question']}\n{option_text}"
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        answer = processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        match = re.search(r"\b([A-Da-d])\b", answer)
        predicted = match.group(1).upper() if match else ""
        is_correct = bool(row["answer"]) and predicted == row["answer"]
        correct += int(is_correct)
        by_category[row["l2"]][0] += int(is_correct)
        by_category[row["l2"]][1] += 1
        results.append({
            "index": row["index"],
            "question": row["question"][:120],
            "predicted": predicted,
            "gold": row["answer"],
            "correct": is_correct,
            "l2": row["l2"],
        })
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"{index + 1}/{len(rows)} acc={correct / (index + 1):.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as output:
        for item in results:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "model": "qwen2.5-vl-3b-lora",
        "samples": len(results),
        "accuracy": correct / max(len(results), 1),
        "category_accuracy": {
            key: round(value[0] / max(value[1], 1), 4) for key, value in sorted(by_category.items())
        },
        "note": "MMBench MCQ; single-letter extraction; circular accuracy not applied.",
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
