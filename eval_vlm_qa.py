"""Evaluate MiniMind-V on VQAv2 or POPE with deterministic sampling."""

import argparse
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

from eval_vlm_benchmark import load_model
from model.model_vlm import MiniMindVLM
from trainer.trainer_utils import setup_seed


NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}


def normalize_answer(text):
    text = text.lower().strip().replace("\n", " ").replace("\t", " ")
    text = "".join(" " if char in string.punctuation else char for char in text)
    tokens = [NUMBER_WORDS.get(token, token) for token in text.split()]
    tokens = [token for token in tokens if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def vqa_consensus(prediction, answers):
    prediction = normalize_answer(prediction)
    normalized = [normalize_answer(answer) for answer in answers]
    if not normalized:
        return 0.0
    scores = []
    for index in range(len(normalized)):
        matches = sum(
            prediction == answer for other, answer in enumerate(normalized) if other != index
        )
        scores.append(min(1.0, matches / 3.0))
    return sum(scores) / len(scores)


def image_path(image_dir, image_id=None, image_name=None):
    directory = Path(image_dir)
    if image_id is not None:
        candidate = directory / f"{int(image_id):012d}.jpg"
        if candidate.is_file():
            return candidate
    if image_name:
        candidate = directory / image_name
        if candidate.is_file():
            return candidate
        match = re.search(r"(\d{12})", image_name)
        if match:
            candidate = directory / f"{match.group(1)}.jpg"
            if candidate.is_file():
                return candidate
    return None


def load_vqav2(questions_path, annotations_path, image_dir, max_samples, seed):
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))["questions"]
    annotations = {
        item["question_id"]: item
        for item in json.loads(Path(annotations_path).read_text(encoding="utf-8"))["annotations"]
    }
    records = []
    for question in questions:
        annotation = annotations.get(question["question_id"])
        path = image_path(image_dir, image_id=question["image_id"])
        if annotation and path:
            records.append({
                "id": question["question_id"],
                "image_path": str(path),
                "question": question["question"],
                "answers": [answer["answer"] for answer in annotation["answers"]],
                "answer_type": annotation["answer_type"],
            })
    random.Random(seed).shuffle(records)
    return records[:max_samples]


def load_pope(path, image_dir, max_samples, seed):
    records = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            resolved = image_path(image_dir, image_name=item.get("image"))
            if resolved:
                records.append({
                    "id": item.get("question_id", len(records)),
                    "image_path": str(resolved),
                    "question": item.get("text", item.get("question")),
                    "label": item["label"].lower(),
                })
    random.Random(seed).shuffle(records)
    return records[:max_samples]


def generate(args, model, tokenizer, processor, record):
    marker = model.config.image_special_token * model.config.image_token_len
    content = f"<image>\nQuestion: {record['question']} Answer with a short phrase."
    if args.task == "pope":
        content = f"<image>\n{record['question']} Answer only yes or no."
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content.replace("<image>", marker)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(args.device)
    image = Image.open(record["image_path"]).convert("RGB")
    pixels = {
        key: value.to(args.device)
        for key, value in MiniMindVLM.image2tensor(image, processor).items()
    }
    with torch.inference_mode():
        output = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            pixel_values=pixels, do_sample=False, max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def summarize_vqa(rows):
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["answer_type"]].append(row["score"])
    return {
        "samples": len(rows),
        "vqa_consensus_accuracy": sum(row["score"] for row in rows) / len(rows),
        "majority_exact_match": sum(row["majority_exact"] for row in rows) / len(rows),
        "by_answer_type": {
            key: sum(values) / len(values) for key, values in sorted(by_type.items())
        },
    }


def summarize_pope(rows):
    tp = sum(row["prediction_label"] == "yes" and row["label"] == "yes" for row in rows)
    fp = sum(row["prediction_label"] == "yes" and row["label"] == "no" for row in rows)
    fn = sum(row["prediction_label"] != "yes" and row["label"] == "yes" for row in rows)
    correct = sum(row["prediction_label"] == row["label"] for row in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "samples": len(rows),
        "accuracy": correct / len(rows),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "yes_ratio": sum(row["prediction_label"] == "yes" for row in rows) / len(rows),
        "unknown_ratio": sum(row["prediction_label"] == "unknown" for row in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description="MiniMind-V VQAv2/POPE evaluation")
    parser.add_argument("--task", choices=["vqav2", "pope"], required=True)
    parser.add_argument("--questions")
    parser.add_argument("--annotations")
    parser.add_argument("--pope_file")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--vision_model_path", default="model/siglip2-base-p32-256-ve")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--projector_type", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    setup_seed(args.seed)
    if args.task == "vqav2":
        if not args.questions or not args.annotations:
            parser.error("vqav2 requires --questions and --annotations")
        records = load_vqav2(
            args.questions, args.annotations, args.image_dir, args.max_samples, args.seed
        )
    else:
        if not args.pope_file:
            parser.error("pope requires --pope_file")
        records = load_pope(args.pope_file, args.image_dir, args.max_samples, args.seed)
    if not records:
        raise RuntimeError("no evaluation records overlap the supplied image directory")
    model, tokenizer, processor = load_model(args)
    rows = []
    for index, record in enumerate(records, 1):
        prediction = generate(args, model, tokenizer, processor, record)
        row = {**record, "prediction": prediction}
        if args.task == "vqav2":
            row["score"] = vqa_consensus(prediction, record["answers"])
            majority = Counter(normalize_answer(answer) for answer in record["answers"]).most_common(1)[0][0]
            row["majority_exact"] = normalize_answer(prediction) == majority
        else:
            normalized = normalize_answer(prediction)
            row["prediction_label"] = (
                "yes" if normalized.startswith("yes") else
                "no" if normalized.startswith("no") else "unknown"
            )
        rows.append(row)
        if index % 50 == 0 or index == len(records):
            print(f"[{args.task}] {index}/{len(records)}", flush=True)
    summary = summarize_vqa(rows) if args.task == "vqav2" else summarize_pope(rows)
    summary.update({"task": args.task, "weight": args.weight, "seed": args.seed})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prediction_path = output.with_name(output.stem + "_predictions.jsonl")
    with open(prediction_path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
