"""POPE-style hallucination evaluation built from COCO2017 val2017.

Follows the POPE protocol (Li et al., 2023): sample images that contain
object annotations; ask "Is there a {object} in the image?" with 3 positive
objects per image and 3 negative objects per difficulty setting:
  - random:      negative objects sampled from categories absent in the image
  - popular:     negative objects sampled from the 10 most frequent categories
  - adversarial: negative objects sampled from categories that co-occur with
                 the image's objects in the dataset but are absent here
The benchmark reports accuracy / precision / recall / F1 / yes-ratio per
setting, which measures object-grounded hallucination rather than caption
lexical overlap.
"""

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description="POPE hallucination evaluation")
    parser.add_argument("--model", choices=["minimind", "qwen3b"], required=True)
    parser.add_argument("--instances_file", default="dataset/coco2017/annotations/instances_val2017.json")
    parser.add_argument("--image_dir", default="dataset/coco2017/val2017")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="sft_full_vlm")
    parser.add_argument("--model_path", default="model/qwen25vl-3b-instruct")
    parser.add_argument("--adapter_path", default="out/qwen_vl_lora", help="empty = zero-shot")
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--pos_per_image", type=int, default=3)
    parser.add_argument("--neg_per_image", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--output_dir", default="results/pope")
    parser.add_argument("--tag", default=None, help="output subdir; default = model")
    parser.add_argument("--constrained", action="store_true",
                        help="restrict first token to yes/no and report calibration (ECE/AUROC)")
    return parser.parse_args()


def parse_yes_no(text):
    text = text.strip().lower()
    if re.match(r"^(yes|yeah|y)", text):
        return "yes"
    if re.match(r"^(no|nope|not)", text):
        return "no"
    return None


def expected_calibration_error(labels, confidences, num_bins=10):
    bins = [[] for _ in range(num_bins)]
    for label, confidence in zip(labels, confidences):
        index = min(int(confidence * num_bins), num_bins - 1)
        bins[index].append((label, confidence))
    total = max(len(labels), 1)
    ece = 0.0
    for bin_items in bins:
        if not bin_items:
            continue
        bin_labels = [item[0] for item in bin_items]
        bin_conf = [item[1] for item in bin_items]
        accuracy = sum(bin_labels) / len(bin_labels)
        ece += len(bin_items) / total * abs(accuracy - sum(bin_conf) / len(bin_conf))
    return ece


def auroc(labels, confidences):
    pairs = sorted(zip(confidences, labels), key=lambda pair: -pair[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    rank_sum = sum(index + 1 for index, (_, label) in enumerate(pairs) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def build_pope_records(instances_file, image_dir, num_images, pos_per_image, neg_per_image, seed):
    with open(instances_file, encoding="utf-8") as file:
        data = json.load(file)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    image_cats = defaultdict(set)
    for ann in data["annotations"]:
        image_cats[ann["image_id"]].add(cat_id_to_name[ann["category_id"]])

    images = sorted(
        (image_id for image_id, cats in image_cats.items()
         if len(cats) >= pos_per_image and (Path(image_dir) / f"{image_id:012d}.jpg").is_file()),
        key=lambda image_id: -len(image_cats[image_id]),
    )
    rng = random.Random(seed)
    rng.shuffle(images)
    selected = images[:num_images]
    if len(selected) < num_images:
        print(f"warning: only {len(selected)} images with >= {pos_per_image} categories")

    all_categories = sorted(cat_id_to_name.values())
    popularity = Counter(cat_id_to_name[ann["category_id"]] for ann in data["annotations"])
    popular_categories = [name for name, _ in popularity.most_common(10)]

    co_occurrence = defaultdict(Counter)
    for ann in data["annotations"]:
        name = cat_id_to_name[ann["category_id"]]
        for other in image_cats[ann["image_id"]]:
            if other != name:
                co_occurrence[name][other] += 1

    records = []
    for image_id in selected:
        present = set(image_cats[image_id])
        positive = rng.sample(sorted(present), min(pos_per_image, len(present)))
        absent = sorted(set(all_categories) - present)
        negative = {
            "random": rng.sample(absent, min(neg_per_image, len(absent))),
            "popular": rng.sample(
                sorted(set(popular_categories) - present), min(neg_per_image, len(set(popular_categories) - present))
            ),
        }
        adversarial_pool = sorted(set().union(
            *(co_occurrence[cat] for cat in present)
        ) - present) if present else []
        negative["adversarial"] = rng.sample(adversarial_pool, min(neg_per_image, len(adversarial_pool)))

        for obj in positive:
            records.append({
                "image_id": image_id, "image_path": str(Path(image_dir) / f"{image_id:012d}.jpg"),
                "object": obj, "label": "yes", "setting": "shared",
            })
        for setting, objects in negative.items():
            for obj in objects:
                records.append({
                    "image_id": image_id, "image_path": str(Path(image_dir) / f"{image_id:012d}.jpg"),
                    "object": obj, "label": "no", "setting": setting,
                })
    return records


def main():
    args = parse_args()
    records = build_pope_records(
        args.instances_file, args.image_dir, args.num_images,
        args.pos_per_image, args.neg_per_image, args.seed,
    )
    print(f"evaluating {len(records)} POPE questions "
          f"(pos={sum(r['label'] == 'yes' for r in records)}, "
          f"neg={sum(r['label'] == 'no' for r in records)})")

    if args.model == "qwen3b":
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.model_path, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="bfloat16", device_map="cuda"
        )
        if args.adapter_path:
            model = PeftModel.from_pretrained(model, args.adapter_path)
        model.eval()
    else:
        import eval_vlm_benchmark as evb
        ns = argparse.Namespace(
            save_dir=args.save_dir, weight=args.weight, tokenizer_path="model",
            vision_model_path="model/siglip2-base-p32-256-ve", hidden_size=768,
            num_hidden_layers=8, projector_type="mlp", use_moe=0,
            device="cuda" if torch.cuda.is_available() else "cpu", dtype="float16",
        )
        model, tokenizer, processor = evb.load_model(ns)
        marker = model.config.image_special_token * model.config.image_token_len

    tag = args.tag or args.model
    output_dir = Path(args.output_dir) / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: {"total": 0, "correct": 0, "tp": 0, "fp": 0, "fn": 0, "yes": 0})
    start = time.perf_counter()
    calibration = {"labels": [], "confidences": []}

    active_tokenizer = processor.tokenizer if args.model == "qwen3b" else tokenizer
    yes_variants = [" yes", "Yes", "yes", " YES"]
    no_variants = [" no", "No", "no", " NO"]
    yes_ids = set()
    no_ids = set()
    for variant in yes_variants:
        ids = active_tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            yes_ids.add(ids[0])
    for variant in no_variants:
        ids = active_tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            no_ids.add(ids[0])

    def infer_response(image, question):
        if args.model == "qwen3b":
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            return processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pixel_values = {
            key: value.to("cuda")
            for key, value in evb.MiniMindVLM.image2tensor(image, processor).items()
        }
        content = question.replace("<image>", marker)
        messages = [{"role": "user", "content": content}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                pixel_values=pixel_values, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def infer_constrained(image, question):
        """First-token yes/no probabilities via constrained generation."""
        if args.model == "qwen3b":
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=1, do_sample=False,
                    return_dict_in_generate=True, output_scores=True,
                )
            scores = generated.scores[0][0]
        else:
            pixel_values = {
                key: value.to("cuda")
                for key, value in evb.MiniMindVLM.image2tensor(image, processor).items()
            }
            content = question.replace("<image>", marker)
            messages = [{"role": "user", "content": content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
            with torch.inference_mode():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=pixel_values,
                )
            scores = outputs.logits[0, -1]
        probs = torch.softmax(scores.float(), dim=-1)
        prob_yes = sum(float(probs[token_id]) for token_id in yes_ids if token_id < probs.numel())
        prob_no = sum(float(probs[token_id]) for token_id in no_ids if token_id < probs.numel())
        return prob_yes, prob_no

    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as output:
        for index, record in enumerate(records):
            image = Image.open(record["image_path"]).convert("RGB")
            question = f"Is there a {record['object']} in the image? Answer yes or no."
            constrained_pred = None
            confidence = None
            if args.constrained:
                prob_yes, prob_no = infer_constrained(image, question)
                constrained_pred = "yes" if prob_yes >= prob_no else "no"
                confidence = max(prob_yes, prob_no) / max(prob_yes + prob_no, 1e-9)
                calibration["labels"].append(int(record["label"] == "yes"))
                calibration["confidences"].append(confidence)
                response = constrained_pred
                predicted = constrained_pred
            else:
                response = infer_response(image, question)
                predicted = parse_yes_no(response)
            is_correct = predicted == record["label"]
            key = record["setting"] if record["setting"] != "shared" else "positive"
            s = stats[key]
            s["total"] += 1
            s["yes"] += int(predicted == "yes")
            if is_correct:
                s["correct"] += 1
            if record["label"] == "yes" and predicted == "yes":
                s["tp"] += 1
            elif record["label"] == "no" and predicted == "yes":
                s["fp"] += 1
            elif record["label"] == "yes" and predicted != "yes":
                s["fn"] += 1
            output.write(json.dumps({
                "image_id": record["image_id"], "object": record["object"],
                "label": record["label"], "setting": record["setting"],
                "question": question, "response": response, "predicted": predicted,
                "correct": is_correct,
                "constrained_predicted": constrained_pred if args.constrained else None,
                "confidence": confidence if args.constrained else None,
            }, ensure_ascii=False) + "\n")
            if (index + 1) % 300 == 0 or index + 1 == len(records):
                elapsed = time.perf_counter() - start
                print(f"{index + 1}/{len(records)} ({elapsed:.0f}s)")

    summaries = {}
    for key, s in stats.items():
        precision = s["tp"] / max(s["tp"] + s["fp"], 1)
        recall = s["tp"] / max(s["tp"] + s["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        summaries[key] = {
            "total": s["total"],
            "accuracy": s["correct"] / max(s["total"], 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "yes_ratio": s["yes"] / max(s["total"], 1),
        }
    overall = {
        "total": sum(s["total"] for s in stats.values()),
        "correct": sum(s["correct"] for s in stats.values()),
        "yes": sum(s["yes"] for s in stats.values()),
    }
    summary = {
        "model": args.model,
        "weight": args.weight if args.model == "minimind" else (args.adapter_path or "zero-shot"),
        "num_images": args.num_images,
        "settings": summaries,
        "overall_accuracy": overall["correct"] / max(overall["total"], 1),
        "overall_yes_ratio": overall["yes"] / max(overall["total"], 1),
        "note": "POPE-style object existence probing on COCO2017 val2017 (see script header).",
    }
    if args.constrained:
        labels = calibration["labels"]
        confidences = calibration["confidences"]
        summary["constrained_calibration"] = {
            "ece": expected_calibration_error(labels, confidences),
            "auroc": auroc(labels, confidences),
            "mean_confidence": sum(confidences) / max(len(confidences), 1),
            "base_rate_yes": sum(labels) / max(len(labels), 1),
        }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
