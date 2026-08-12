"""Single-GPU GRPO post-training for native MiniMind-V checkpoints.

The trainer samples a group of answers for each image prompt, assigns an
exact-token reference reward, normalizes rewards within the group, and applies
the clipped GRPO objective with a frozen SFT reference policy and KL penalty.
"""

import argparse
import io
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

__package__ = "trainer"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset.grpo_dataset import VLMGRPODataset, grpo_collate_fn
from evaluation.caption_metrics import tokenize
from evaluation.grpo_rewards import (
    build_document_frequency,
    cider_style_sample,
    reference_reward,
    update_kl_beta,
)
from model.model_vlm import MiniMindVLM, VLMConfig
from model.torch_compat import ensure_torch_transformers_compat
from trainer.trainer_utils import get_model_params, init_vlm_model, setup_seed

ensure_torch_transformers_compat()


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind-V GRPO post-training")
    parser.add_argument("--data_path", default="dataset/grpo_i2t.parquet")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--from_weight", default="sft_vlm")
    parser.add_argument("--save_weight", default="grpo_vlm")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--vision_model_path", default="model/siglip2-base-p32-256-ve")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--freeze_llm", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, choices=[1], default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--adaptive_beta", type=int, choices=[0, 1], default=1)
    parser.add_argument("--target_kl", type=float, default=0.10)
    parser.add_argument("--beta_update_rate", type=float, default=0.05)
    parser.add_argument("--min_beta", type=float, default=1e-4)
    parser.add_argument("--max_beta", type=float, default=2.0)
    parser.add_argument("--kl_stop", type=float, default=1.0, help="0 disables the KL safety stop")
    parser.add_argument("--kl_stop_patience", type=int, default=20)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_prompt_length", type=int, default=384)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--reward_unigram", type=float, default=0.30)
    parser.add_argument("--reward_rouge", type=float, default=0.30)
    parser.add_argument("--reward_meteor", type=float, default=0.30)
    parser.add_argument("--reward_length", type=float, default=0.10)
    parser.add_argument("--reward_repetition", type=float, default=0.20)
    parser.add_argument("--reward_cider", type=float, default=0.0, help="weight of batch-level CIDEr-style reward")
    parser.add_argument("--reward_judge", type=float, default=0.0,
                        help="weight of a stronger-model judge reward (Qwen2.5-VL)")
    parser.add_argument("--judge_model_path", default="model/qwen25vl-3b-instruct")
    parser.add_argument("--judge_adapter_path", default="",
                        help="optional LoRA adapter for the judge (empty = base instruct model)")
    parser.add_argument("--judge_max_new_tokens", type=int, default=4)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means the full dataset")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=int, choices=[0, 1], default=0)
    parser.add_argument("--log_file", default="logs/grpo_train.jsonl")
    return parser.parse_args()


def repeat_pixels(pixel_values, count, device):
    if hasattr(pixel_values, "keys"):
        return {
            key: value.to(device, non_blocking=True).repeat_interleave(count, dim=0)
            for key, value in pixel_values.items()
        }
    return pixel_values.to(device, non_blocking=True).repeat_interleave(count, dim=0)


def completion_mask(completion_ids, eos_token_id):
    mask = torch.ones_like(completion_ids, dtype=torch.float32)
    if eos_token_id is None:
        return mask
    for row in range(completion_ids.size(0)):
        positions = torch.nonzero(completion_ids[row].eq(eos_token_id), as_tuple=False)
        if positions.numel():
            mask[row, positions[0, 0] + 1:] = 0
    return mask


def sequence_log_probs(model, sequences, prompt_length, pixel_values, device_type, amp_dtype):
    amp_context = (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if device_type == "cuda" else nullcontext()
    )
    with amp_context:
        logits = model(input_ids=sequences, pixel_values=pixel_values).logits
        completion_logits = logits[:, prompt_length - 1:-1, :]
        completion_ids = sequences[:, prompt_length:]
        return F.log_softmax(completion_logits.float(), dim=-1).gather(
            -1, completion_ids.unsqueeze(-1)
        ).squeeze(-1)


def save_training_state(args, model, optimizer, epoch, step, config):
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    suffix = "_moe" if config.use_moe else ""
    state = {
        key: value.detach().half().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("vision_encoder.")
    }
    weight_path = Path(args.save_dir) / f"{args.save_weight}_{config.hidden_size}{suffix}.pth"
    temp_path = weight_path.with_suffix(weight_path.suffix + ".tmp")
    torch.save(state, temp_path)
    os.replace(temp_path, weight_path)
    resume_path = Path(args.checkpoint_dir) / f"{args.save_weight}_{config.hidden_size}{suffix}_resume.pth"
    resume_temp = resume_path.with_suffix(resume_path.suffix + ".tmp")
    torch.save({
        "model": state,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": vars(args),
    }, resume_temp)
    os.replace(resume_temp, resume_path)
    return weight_path, resume_path


def main():
    args = parse_args()
    setup_seed(args.seed)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_prompt_length + args.max_new_tokens,
        use_moe=bool(args.use_moe),
    )
    policy, tokenizer, processor = init_vlm_model(
        config,
        from_weight=args.from_weight,
        tokenizer_path=args.tokenizer_path,
        vision_model_path=args.vision_model_path,
        save_dir=args.save_dir,
        device=args.device,
        freeze_llm=args.freeze_llm,
    )
    suffix = "_moe" if args.use_moe else ""
    source_path = Path(args.save_dir) / f"{args.from_weight}_{args.hidden_size}{suffix}.pth"
    source_state = torch.load(source_path, map_location="cpu", weights_only=True)
    reference = MiniMindVLM(config, vision_model_path=args.vision_model_path)
    reference.load_state_dict(source_state, strict=False)
    reference.requires_grad_(False).eval().to(args.device)
    policy.vision_encoder.eval()
    get_model_params(reference, config)

    judge_model = judge_processor = None
    if args.reward_judge > 0:
        from PIL import Image
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        print(f"loading judge model {args.judge_model_path} ...", flush=True)
        judge_processor = AutoProcessor.from_pretrained(
            args.judge_model_path, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
        )
        judge_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.judge_model_path, torch_dtype="bfloat16", device_map="cuda"
        )
        if args.judge_adapter_path:
            from peft import PeftModel
            judge_model = PeftModel.from_pretrained(judge_model, args.judge_adapter_path)
        judge_model.eval()

    def judge_scores_for(completions, image_bytes):
        """Score each completion 1..5 with the stronger judge model, mapped to [-1, 1]."""
        from PIL import Image as PILImage
        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        scores = []
        for text in completions:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        "Rate this image description from 1 to 5 for factual accuracy, "
                        f"completeness and naturalness. Output only the integer score.\nDescription: {text}"
                    )},
                ],
            }]
            prompt = judge_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = judge_processor(text=[prompt], images=[image], return_tensors="pt").to(args.device)
            with torch.inference_mode():
                generated = judge_model.generate(
                    **inputs, max_new_tokens=args.judge_max_new_tokens, do_sample=False
                )
            response = judge_processor.decode(
                generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            match = re.search(r"\d+(\.\d+)?", response)
            parsed = float(match.group(0)) if match else 3.0
            scores.append(max(-1.0, min(1.0, (parsed - 3.0) / 2.0)))
        return scores

    dataset = VLMGRPODataset(
        args.data_path,
        tokenizer,
        processor,
        max_prompt_length=args.max_prompt_length,
        image_special_token=config.image_special_token,
        image_token_len=config.image_token_len,
    )
    document_frequency = None
    document_count = 0
    if args.reward_cider > 0:
        print("building CIDEr document frequency over training references...", flush=True)
        document_frequency, document_count = build_document_frequency(
            [[tokenize(text)] for refs in dataset.references() for text in refs]
        )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=grpo_collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    optimizer = AdamW(filter(lambda parameter: parameter.requires_grad, policy.parameters()), lr=args.learning_rate)
    start_epoch = start_step = 0
    resume_path = Path(args.checkpoint_dir) / f"{args.save_weight}_{args.hidden_size}{suffix}_resume.pth"
    if args.resume and resume_path.exists():
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        policy.load_state_dict(resume["model"], strict=False)
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch, start_step = resume["epoch"], resume["step"]
        args.beta = resume.get("args", {}).get("beta", args.beta)
        print(f"resumed from epoch={start_epoch}, step={start_step}", flush=True)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    global_step = 0
    kl_violations = 0
    stopped_for_kl = False
    reward_weights = {
        "unigram_f1": args.reward_unigram,
        "rouge_l": args.reward_rouge,
        "meteor_exact": args.reward_meteor,
        "length_score": args.reward_length,
        "repetition_penalty": args.reward_repetition,
    }
    if args.reward_cider > 0:
        reward_weights["cider_style"] = args.reward_cider
    started = time.time()
    log_mode = "a" if args.resume else "w"
    with open(args.log_file, log_mode, encoding="utf-8") as log_output:
        for epoch in range(start_epoch, args.epochs):
            for epoch_step, batch in enumerate(loader, 1):
                if epoch == start_epoch and epoch_step <= start_step:
                    continue
                global_step += 1
                input_ids = batch["input_ids"].to(args.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(args.device, non_blocking=True)
                pixels_one = {
                    key: value.to(args.device, non_blocking=True)
                    for key, value in batch["pixel_values"].items()
                }
                policy.eval()
                with torch.inference_mode():
                    sequences = policy.generate(
                        inputs=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixels_one,
                        num_return_sequences=args.group_size,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                # Tensors created inside inference_mode cannot be captured by
                # autograd during the policy update. Materialize a normal
                # tensor while preserving the sampled token ids exactly.
                sequences = sequences.clone()
                prompt_length = input_ids.size(1)
                completion_ids = sequences[:, prompt_length:]
                token_mask = completion_mask(completion_ids, tokenizer.eos_token_id).to(args.device)
                completions = [
                    tokenizer.decode(
                        ids[token_mask[row].bool()].detach().cpu().tolist(),
                        skip_special_tokens=True,
                    ).strip()
                    for row, ids in enumerate(completion_ids)
                ]
                judge_scores = []
                if args.reward_judge > 0:
                    judge_scores = judge_scores_for(completions, batch["image_bytes"])
                if args.reward_cider > 0:
                    reference_tokens = [tokenize(batch["reference"])]
                    reward_items = []
                    for position, text in enumerate(completions):
                        base, components = reference_reward(
                            text, [batch["reference"]], weights=reward_weights
                        )
                        cider_value = cider_style_sample(
                            tokenize(text), reference_tokens, document_frequency, document_count
                        ) / 10.0
                        components["cider_style"] = cider_value
                        total = base + args.reward_cider * cider_value
                        if judge_scores:
                            components["judge_score"] = judge_scores[position]
                            total = total + args.reward_judge * judge_scores[position]
                        reward_items.append((max(-1.0, min(1.0, total)), components))
                else:
                    reward_items = []
                    for position, text in enumerate(completions):
                        base, components = reference_reward(
                            text, [batch["reference"]], weights=reward_weights
                        )
                        total = base
                        if judge_scores:
                            components["judge_score"] = judge_scores[position]
                            total = base + args.reward_judge * judge_scores[position]
                        reward_items.append((max(-1.0, min(1.0, total)), components))
                rewards = torch.tensor([item[0] for item in reward_items], device=args.device)
                advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
                pixels_group = repeat_pixels(batch["pixel_values"], args.group_size, args.device)
                policy.eval()
                with torch.no_grad():
                    old_log_probs = sequence_log_probs(
                        policy, sequences, prompt_length, pixels_group, device_type, amp_dtype
                    )
                    reference_log_probs = sequence_log_probs(
                        reference, sequences, prompt_length, pixels_group, device_type, amp_dtype
                    )

                losses, kls, clip_fractions = [], [], []
                for _ in range(args.ppo_epochs):
                    policy.train()
                    policy.vision_encoder.eval()
                    new_log_probs = sequence_log_probs(
                        policy, sequences, prompt_length, pixels_group, device_type, amp_dtype
                    )
                    log_ratio = new_log_probs - old_log_probs
                    ratio = torch.exp(log_ratio)
                    advantage = advantages[:, None]
                    unclipped = ratio * advantage
                    clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * advantage
                    policy_loss = -torch.minimum(unclipped, clipped)
                    log_ref_ratio = reference_log_probs - new_log_probs
                    kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1
                    loss = ((policy_loss + args.beta * kl) * token_mask).sum() / token_mask.sum().clamp_min(1)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                    optimizer.step()
                    losses.append(loss.detach().item())
                    kls.append(((kl * token_mask).sum() / token_mask.sum().clamp_min(1)).detach().item())
                    clip_fractions.append(
                        ((((ratio - 1).abs() > args.clip_eps).float() * token_mask).sum()
                         / token_mask.sum().clamp_min(1)).detach().item()
                    )

                observed_kl = sum(kls) / len(kls)
                reward_components = {
                    key: sum(item[1][key] for item in reward_items) / len(reward_items)
                    for key in reward_items[0][1]
                }
                if args.adaptive_beta:
                    args.beta = update_kl_beta(
                        args.beta, observed_kl, args.target_kl,
                        rate=args.beta_update_rate,
                        minimum=args.min_beta, maximum=args.max_beta,
                    )
                kl_violations = kl_violations + 1 if args.kl_stop and observed_kl > args.kl_stop else 0
                record = {
                    "epoch": epoch + 1,
                    "epoch_step": epoch_step,
                    "global_step": global_step,
                    "loss": sum(losses) / len(losses),
                    "reward_mean": rewards.mean().item(),
                    "reward_std": rewards.std(unbiased=False).item(),
                    "kl": observed_kl,
                    "beta": args.beta,
                    "clip_fraction": sum(clip_fractions) / len(clip_fractions),
                    "reward_components": reward_components,
                    "completion_tokens": token_mask.sum(dim=1).float().mean().item(),
                    "elapsed_seconds": time.time() - started,
                    "row_index": batch["row_index"],
                }
                log_output.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_output.flush()
                if global_step % args.log_interval == 0 or global_step == 1:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                if global_step % args.save_interval == 0:
                    paths = save_training_state(args, policy, optimizer, epoch, epoch_step, config)
                    print(f"saved: {paths[0]} and {paths[1]}", flush=True)
                if args.kl_stop and kl_violations >= args.kl_stop_patience:
                    print(
                        f"KL safety stop: kl>{args.kl_stop} for "
                        f"{args.kl_stop_patience} consecutive steps",
                        flush=True,
                    )
                    stopped_for_kl = True
                    break
                if args.max_steps and global_step >= args.max_steps:
                    break
            start_step = 0
            if stopped_for_kl or (args.max_steps and global_step >= args.max_steps):
                break

    paths = save_training_state(args, policy, optimizer, epoch, epoch_step, config)
    print(f"training complete; saved: {paths[0]} and {paths[1]}", flush=True)


if __name__ == "__main__":
    main()
