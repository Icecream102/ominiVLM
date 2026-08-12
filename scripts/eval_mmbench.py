"""MMBench (en/dev) multiple-choice evaluation for MiniMind-V and Qwen2.5-VL LoRA."""

import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description="MMBench en/dev evaluation")
    parser.add_argument("--dataset_dir", default="dataset/mmbench_en_dev")
    parser.add_argument("--model", choices=["qwen3b", "minimind"], required=True)
    parser.add_argument("--model_path", default="model/qwen25vl-3b-instruct")
    parser.add_argument("--adapter_path", default="out/qwen_vl_lora", help="empty = zero-shot (no LoRA)")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="vqa_sft_mix_vlm")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results/mmbench")
    parser.add_argument("--tag", default="qwen3b", help="output subdirectory name")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    return parser.parse_args()


def build_options(example):
    lines = [f"{example['question']}"]
    letters = ["A", "B", "C", "D"]
    valid = [(letter, example[letter]) for letter in letters if str(example.get(letter)) != "nan"]
    lines.append("Options:")
    for letter, option in valid:
        lines.append(f"{letter}. {option}")
    lines.append("Answer with the option letter only.")
    return "\n".join(lines), valid


def extract_answer(text):
    text = text.strip()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"answer\s*[:\-]?\s*([ABCD])", text, re.IGNORECASE)
    return match.group(1) if match else None


def load_minimind(args):
    import eval_vlm_benchmark as evb
    from model.model_vlm import MiniMindVLM
    ns = argparse.Namespace(
        save_dir=args.save_dir, weight=args.weight, tokenizer_path="model",
        vision_model_path="model/siglip2-base-p32-256-ve", hidden_size=768,
        num_hidden_layers=8, projector_type="mlp", use_moe=0,
        device="cuda" if torch.cuda.is_available() else "cpu", dtype="float16",
    )
    model, tokenizer, processor = evb.load_model(ns)
    return model, tokenizer, processor, MiniMindVLM


def main():
    args = parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_from_disk(args.dataset_dir)
    indices = list(range(len(data)))
    random.Random(args.seed).shuffle(indices)
    if args.max_samples:
        indices = indices[: args.max_samples]
    print(f"evaluating {len(indices)} MMBench en/dev questions")

    if args.model == "qwen3b":
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype="bfloat16", device_map="cuda"
        )
        if args.adapter_path:
            model = PeftModel.from_pretrained(model, args.adapter_path)
        model.eval()
    else:
        model, tokenizer, processor, vlm_cls = load_minimind(args)
        marker = model.config.image_special_token * model.config.image_token_len

    output_dir = Path(args.output_dir) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    correct = 0
    start = time.perf_counter()
    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as output:
        for position, index in enumerate(indices):
            example = data[int(index)]
            image = example["image"].convert("RGB")
            question_text, valid = build_options(example)
            answer = example["answer"]
            if args.model == "qwen3b":
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question_text},
                    ],
                }]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                response = processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            else:
                content = question_text.replace("<image>", marker)
                messages = [{"role": "user", "content": content}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(args.device)
                pixel_values = {
                    key: value.to(args.device)
                    for key, value in vlm_cls.image2tensor(image, processor).items()
                }
                with torch.inference_mode():
                    generated = model.generate(
                        inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                        pixel_values=pixel_values, max_new_tokens=args.max_new_tokens,
                        do_sample=False, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                response = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            predicted = extract_answer(response)
            is_correct = predicted == answer and answer in [letter for letter, _ in valid]
            correct += int(is_correct)
            output.write(json.dumps({
                "index": index, "question": example["question"], "category": example.get("category"),
                "options": [f"{letter}. {option}" for letter, option in valid],
                "answer": answer, "response": response, "predicted": predicted, "correct": is_correct,
            }, ensure_ascii=False) + "\n")
            if (position + 1) % 100 == 0 or position + 1 == len(indices):
                elapsed = time.perf_counter() - start
                print(f"{position + 1}/{len(indices)} acc={correct / (position + 1):.4f} ({elapsed:.0f}s)")

    summary = {
        "model": args.model,
        "samples": len(indices),
        "accuracy": correct / max(len(indices), 1),
        "note": "MMBench en/dev single-pass MCQ accuracy (no circular evaluation).",
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
