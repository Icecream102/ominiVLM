"""Text-only robustness evaluation.

Runs VQA-style questions WITHOUT an image and measures how the model behaves:
  - refusal/hedge rate ("As an AI...", "I cannot", "Based on the image..."-style
    hedging when there is no image is a hallucination risk);
  - answer agreement with the reference (should be near chance / low);
  - empty-response rate.
The point is to expose whether the model hallucinates content for missing
visual input instead of refusing or saying it cannot see the image.
"""

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_vlm_benchmark as evb


REFUSAL_PATTERNS = [
    r"as an ai",
    r"i (can'?t|cannot|am not able)",
    r"i don'?t have",
    r"cannot see",
    r"no image",
    r"without (the|an) image",
    r"unable to",
    r"not (able to|possible) (to )?(determine|answer|see|provide)",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Text-only robustness eval")
    parser.add_argument("--questions_file", default="dataset/vqav2/v2_OpenEnded_mscoco_val2014_questions.json")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="multitask_final_vlm")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--output_dir", default="results/text_only")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.questions_file, encoding="utf-8") as file:
        questions = json.load(file)["questions"]
    random.Random(args.seed).shuffle(questions)
    questions = questions[: args.max_samples]

    import eval_vlm_benchmark as evb
    ns = argparse.Namespace(
        save_dir=args.save_dir, weight=args.weight, tokenizer_path="model",
        vision_model_path="model/siglip2-base-p32-256-ve", hidden_size=768,
        num_hidden_layers=8, projector_type="mlp", use_moe=0,
        device="cuda" if torch.cuda.is_available() else "cpu", dtype="float16",
    )
    model, tokenizer, processor = evb.load_model(ns)

    output_dir = Path(args.output_dir) / args.weight
    output_dir.mkdir(parents=True, exist_ok=True)
    refusal = 0
    empty = 0
    hallucinated_content = 0
    start = time.perf_counter()
    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as output:
        for index, question in enumerate(questions):
            messages = [{
                "role": "user",
                "content": f"{question['question']} Answer in one word or a short phrase.",
            }]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(ns.device)
            with torch.inference_mode():
                generated = model.generate(
                    inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
            answer = tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip().lower()
            is_refusal = bool(re.search("|".join(REFUSAL_PATTERNS), answer))
            is_empty = not answer
            # A concrete factual claim (not refusal/empty) without image = hallucination risk.
            is_hedge = bool(re.search(r"(maybe|perhaps|possibly|could be|might be)", answer))
            if not is_refusal and not is_empty and not is_hedge:
                hallucinated_content += 1
            refusal += int(is_refusal)
            empty += int(is_empty)
            output.write(json.dumps({
                "question": question["question"],
                "answer": answer,
                "is_refusal": is_refusal,
                "is_empty": is_empty,
                "is_hedge": is_hedge,
            }, ensure_ascii=False) + "\n")
            if (index + 1) % 100 == 0 or index + 1 == len(questions):
                elapsed = time.perf_counter() - start
                print(f"{index + 1}/{len(questions)} ({elapsed:.0f}s)")

    total = max(len(questions), 1)
    summary = {
        "weight": args.weight,
        "samples": len(questions),
        "refusal_rate": refusal / total,
        "empty_rate": empty / total,
        "concrete_answer_rate": hallucinated_content / total,
        "note": "Text-only questions (no image); concrete answers to unanswerable questions "
                "are treated as hallucination risk.",
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
