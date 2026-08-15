"""DPO for Qwen2.5-VL LoRA on judge-built preference pairs (single GPU).

Manual DPO implementation (trl's DPOTrainer has limited vision support):
policy = 4-bit QLoRA model, reference = frozen base model. Loss:
  -log sigmoid(beta * (log_pi(chosen) - log_pi(rejected)
                       - log_ref(chosen) + log_ref(rejected)))
"""

import argparse
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from peft import PeftModel
from torch.optim import AdamW
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)


def parse_args():
    parser = argparse.ArgumentParser(description="DPO for Qwen2.5-VL LoRA")
    parser.add_argument("--model_path", default="model/qwen25vl-3b-instruct")
    parser.add_argument("--adapter_path", default="", help="optional LoRA base to continue from")
    parser.add_argument("--data_path", default="dataset/preference_pairs.parquet")
    parser.add_argument("--output_dir", default="out/qwen_vl_dpo")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--kl_lambda", type=float, default=0.1, help="reference-KL regularization weight")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=0)
    return parser.parse_args()


def load_pairs(parquet_path, limit=0):
    table = pq.read_table(parquet_path)
    pairs = []
    for index in range(table.num_rows):
        image_bytes = table.column("image_bytes")[index].as_py()
        if isinstance(image_bytes, list):
            image_bytes = image_bytes[0]
        pairs.append({
            "image_bytes": image_bytes,
            "prompt": table.column("prompt")[index].as_py(),
            "chosen": table.column("chosen")[index].as_py(),
            "rejected": table.column("rejected")[index].as_py(),
        })
    return pairs[:limit] if limit else pairs


def build_inputs(processor, image, prompt, response):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": response},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    encoded = processor(text=text, images=image, return_tensors="pt")
    prompt_messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_encoded = processor(text=prompt_text, images=image, return_tensors="pt")
    labels = encoded["input_ids"].clone()
    labels[:, : prompt_encoded["input_ids"].shape[1]] = -100
    pixel_values = encoded["pixel_values"]
    if isinstance(pixel_values, (list, tuple)):
        pixel_values = torch.cat(
            [p if p.dim() == 4 else p.squeeze(0) for p in pixel_values], dim=0
        )
    elif pixel_values.dim() == 5:
        pixel_values = pixel_values[0]
    grid_thw = encoded["image_grid_thw"]
    if isinstance(grid_thw, (list, tuple)):
        grid_thw = torch.stack(grid_thw)
    if grid_thw.dim() == 1:
        grid_thw = grid_thw.unsqueeze(0)
    return {
        "input_ids": encoded["input_ids"],
        "labels": labels,
        "attention_mask": encoded["attention_mask"],
        "pixel_values": pixel_values,
        "image_grid_thw": grid_thw,
    }


def log_prob(model, inputs, device, grad=False):
    context = torch.enable_grad() if grad else torch.inference_mode()
    with context:
        outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            pixel_values={key: value.to(device) for key, value in inputs["pixel_values"].items()}
            if isinstance(inputs["pixel_values"], dict) else inputs["pixel_values"].to(device),
            image_grid_thw=inputs["image_grid_thw"].to(device),
            labels=inputs["input_ids"].to(device),
        )
    logits = outputs.logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    labels = inputs["labels"].to(device) if "labels" in inputs else inputs["input_ids"].to(device)
    indices = inputs["input_ids"].to(device).unsqueeze(-1)
    gathered = log_probs.gather(-1, indices).squeeze(-1)
    mask = (labels != -100).float()
    if not mask.any():
        mask = torch.ones_like(labels, dtype=torch.float)
    return (gathered * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    pairs = load_pairs(args.data_path, args.max_samples)
    random.shuffle(pairs)
    print(f"loaded {len(pairs)} preference pairs")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)
    policy = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    reference = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter_path:
        policy = PeftModel.from_pretrained(policy, args.adapter_path)
        reference = PeftModel.from_pretrained(reference, args.adapter_path)
    for name, parameter in policy.named_parameters():
        parameter.requires_grad = "lora_" in name.lower()
    reference.requires_grad_(False).eval()
    policy.train()
    policy.print_trainable_parameters()

    optimizer = AdamW(filter(lambda p: p.requires_grad, policy.parameters()), lr=args.lr)
    device = "cuda"
    steps = 0
    total_loss = 0.0
    for epoch in range(args.epochs):
        for index in range(0, len(pairs), args.batch_size):
            batch = pairs[index:index + args.batch_size]
            loss_terms = []
            for pair in batch:
                image = Image.open(io.BytesIO(pair["image_bytes"])).convert("RGB")
                chosen_inputs = build_inputs(processor, image, pair["prompt"], pair["chosen"])
                rejected_inputs = build_inputs(processor, image, pair["prompt"], pair["rejected"])
                with torch.no_grad():
                    ref_chosen = log_prob(reference, chosen_inputs, device, grad=False)
                    ref_rejected = log_prob(reference, rejected_inputs, device, grad=False)
                policy.train()
                pi_chosen = log_prob(policy, chosen_inputs, device, grad=True)
                pi_rejected = log_prob(policy, rejected_inputs, device, grad=True)
                log_ratio = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
                kl_penalty = args.kl_lambda * torch.clamp_min(ref_chosen.detach() - pi_chosen, 0.0)
                loss_terms.append(
                    -torch.nn.functional.logsigmoid(args.beta * log_ratio).mean() + kl_penalty.mean()
                )
            loss = torch.stack(loss_terms).mean() / args.grad_accum
            loss.backward()
            total_loss += loss.item() * args.grad_accum
            if (index // args.batch_size + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, policy.parameters()), 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if steps % args.log_interval == 0:
                    print(f"step {steps} loss={total_loss / args.log_interval:.4f}", flush=True)
                    total_loss = 0.0
                if args.max_steps and steps >= args.max_steps:
                    print(f"reached max_steps {args.max_steps}")
                    policy.save_pretrained(args.output_dir)
                    processor.save_pretrained(args.output_dir)
                    print(f"saved DPO adapter to {args.output_dir}")
                    return
    policy.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"saved DPO adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
