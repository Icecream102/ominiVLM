"""Build offline preference pairs with a stronger-model judge (Qwen2.5-VL-3B).

For each prompt/image: sample two responses from the policy, score both with
the judge (1..5), keep the pair only when the score gap >= gap_threshold.
Output parquet: image_bytes, prompt, chosen, rejected — ready for DPO.
"""

import argparse
import io
import json
import random
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Build judge preference pairs")
    parser.add_argument("--data_path", default="dataset/vqa_sft.parquet",
                        help="base parquet (image_bytes + conversations)")
    parser.add_argument("--policy_path", default="model/qwen25vl-3b-instruct")
    parser.add_argument("--policy_adapter", default="", help="LoRA adapter for the policy")
    parser.add_argument("--judge_path", default="model/qwen25vl-3b-instruct")
    parser.add_argument("--judge_adapter", default="", help="LoRA adapter for the judge")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--gap_threshold", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="dataset/preference_pairs.parquet")
    return parser.parse_args()


def main():
    args = parse_args()
    from peft import PeftModel
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    table = pq.read_table(args.data_path, columns=["image_bytes", "conversations"])
    rows = []
    for index in range(table.num_rows):
        conversations = json.loads(table.column("conversations")[index].as_py())
        user_turn = next((t for t in conversations if t.get("role") == "user"), None)
        if user_turn is None:
            continue
        rows.append({
            "image_bytes": table.column("image_bytes")[index].as_py(),
            "prompt": user_turn.get("content", "").replace("<image>", "").strip(),
        })
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.max_samples]
    print(f"candidate rows: {len(rows)}")

    processor = AutoProcessor.from_pretrained(args.policy_path, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    policy = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.policy_path, quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    if args.policy_adapter:
        policy = PeftModel.from_pretrained(policy, args.policy_adapter)
    policy.eval()

    judge = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.judge_path, torch_dtype="bfloat16", device_map="cuda"
    )
    if args.judge_adapter:
        judge = PeftModel.from_pretrained(judge, args.judge_adapter)
    judge.eval()

    def generate(image, prompt, temperature):
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
            generated = policy.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=temperature, top_p=args.top_p,
            )
        return processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def judge_score(image, prompt, response):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "Rate this image description from 1 to 5 for factual accuracy, "
                    f"completeness and naturalness. Output only the integer score.\n"
                    f"Description: {response}"
                )},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = judge.generate(**inputs, max_new_tokens=4, do_sample=False)
        output = processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        match = re.search(r"\d+(\.\d+)?", output)
        return float(match.group(0)) if match else 3.0

    kept = 0
    attempts = 0
    image_bytes_list, prompt_list, chosen_list, rejected_list = [], [], [], []
    start = time.perf_counter()
    for index, row in enumerate(rows):
        image = Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")
        response_a = generate(image, row["prompt"], args.temperature)
        response_b = generate(image, row["prompt"], args.temperature)
        score_a = judge_score(image, row["prompt"], response_a)
        score_b = judge_score(image, row["prompt"], response_b)
        attempts += 1
        if abs(score_a - score_b) >= args.gap_threshold:
            if score_a >= score_b:
                chosen, rejected = response_a, response_b
            else:
                chosen, rejected = response_b, response_a
            image_bytes_list.append(row["image_bytes"])
            prompt_list.append(row["prompt"])
            chosen_list.append(chosen)
            rejected_list.append(rejected)
            kept += 1
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"{index + 1}/{len(rows)} kept={kept} ({time.perf_counter() - start:.0f}s)")

    table_out = pa.table({
        "image_bytes": pa.array(image_bytes_list, type=pa.binary()),
        "prompt": pa.array(prompt_list, type=pa.string()),
        "chosen": pa.array(chosen_list, type=pa.string()),
        "rejected": pa.array(rejected_list, type=pa.string()),
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table_out, args.output, compression="snappy")
    print(f"saved {args.output} with {kept} pairs ({kept / max(attempts, 1):.2f} keep rate)")


if __name__ == "__main__":
    main()
