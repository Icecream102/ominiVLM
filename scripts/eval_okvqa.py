"""OK-VQA open-ended evaluation for MiniMind-V checkpoints.

OK-VQA shares the COCO2014 val2014 image set and the 3-of-10 answer
agreement rule with VQAv2, but tests knowledge-based questions. Annotation
answers are plain strings (not VQAv2-style answer dicts); both formats are
accepted for robustness.
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
    parser = argparse.ArgumentParser(description="OK-VQA evaluation")
    parser.add_argument("--model", choices=["minimind", "qwen3b"], default="minimind")
    parser.add_argument("--questions_file", required=True)
    parser.add_argument("--annotations_file", required=True)
    parser.add_argument("--image_zip", required=True)
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="sft_full_vlm")
    parser.add_argument("--model_path", default="model/qwen25vl-3b-instruct")
    parser.add_argument("--adapter_path", default="out/qwen_vl_lora", help="empty = zero-shot")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--vision_model_path", default="model/siglip2-base-p32-256-ve")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--projector_type", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all questions")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output_dir", default="results/okvqa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default=None, help="output subdir override")
    parser.add_argument("--short_instruction", default=" Answer in one word or a short phrase.")
    return parser.parse_args()


def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_questions_and_answers(questions_file, annotations_file):
    with open(questions_file, encoding="utf-8") as file:
        questions = json.load(file)["questions"]
    with open(annotations_file, encoding="utf-8") as file:
        annotations = json.load(file)["annotations"]
    answers_by_question = defaultdict(list)
    for item in annotations:
        raw_answers = item.get("answers", [])
        normalized = []
        for answer in raw_answers:
            text = answer["answer"] if isinstance(answer, dict) else str(answer)
            normalized.append(normalize_answer(text))
        answers_by_question[item["question_id"]] = normalized
    records = [
        {
            "question_id": q["question_id"],
            "image_id": q["image_id"],
            "question": q["question"],
            "human_answers": answers_by_question.get(q["question_id"], []),
        }
        for q in questions
        if q["question_id"] in answers_by_question
    ]
    return records


def image_loader_factory(zip_path):
    archive = zipfile.ZipFile(zip_path)
    by_id = {}
    for name in archive.namelist():
        match = re.search(r"(\d{12})\.jpg$", name)
        if match:
            by_id[int(match.group(1))] = name

    def load(image_id):
        member = by_id.get(image_id)
        if member is None:
            raise FileNotFoundError(f"image {image_id} not found in zip")
        return Image.open(archive.open(member)).convert("RGB")

    return load


def main():
    args = parse_args()
    records = load_questions_and_answers(args.questions_file, args.annotations_file)
    if args.max_samples:
        random.Random(args.seed).shuffle(records)
        records = records[: args.max_samples]
    print(f"evaluating {len(records)} OK-VQA questions")

    load_image = image_loader_factory(args.image_zip)
    if args.model == "qwen3b":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import eval_vlm_benchmark as evb
        evb.setup_seed(args.seed)
        model, tokenizer, processor = evb.load_model(args)
        marker = model.config.image_special_token * model.config.image_token_len

    tag = args.tag or ("qwen2.5-vl-3b-lora" if args.model == "qwen3b" else args.weight)
    output_dir = Path(args.output_dir) / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    correct = 0
    start = time.perf_counter()
    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as output:
        for index, record in enumerate(records):
            image = load_image(record["image_id"])
            if args.model == "qwen3b":
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": record["question"] + args.short_instruction},
                    ],
                }]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                answer = processor.decode(
                    generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
            else:
                pixel_values = {
                    key: value.to(args.device)
                    for key, value in evb.MiniMindVLM.image2tensor(image, processor).items()
                }
                content = record["question"].replace("<image>", marker) + args.short_instruction
                messages = [{"role": "user", "content": content}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(args.device)
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
                new_ids = generated[0, inputs["input_ids"].shape[1]:]
                answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            normalized = normalize_answer(answer)
            matches = sum(1 for human in record["human_answers"] if human == normalized)
            score = min(1.0, matches / 3)
            correct += score
            predictions = {
                "question_id": record["question_id"],
                "image_id": record["image_id"],
                "question": record["question"],
                "answer": answer,
                "normalized_answer": normalized,
                "matches": matches,
                "score": score,
            }
            output.write(json.dumps(predictions, ensure_ascii=False) + "\n")
            if (index + 1) % 100 == 0 or index + 1 == len(records):
                elapsed = time.perf_counter() - start
                print(f"{index + 1}/{len(records)} acc={correct / (index + 1):.4f} ({elapsed:.0f}s)")

    summary = {
        "weight": tag,
        "samples": len(records),
        "accuracy": correct / max(len(records), 1),
        "note": "OK-VQA val2014; official 3-of-10 agreement rule with answer normalization.",
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
