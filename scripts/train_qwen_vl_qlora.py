"""QLoRA multi-task SFT of Qwen2.5-VL-7B on a single 24GB GPU.

Reads one or more parquet files in the repository's standard schema
(image_bytes + conversations). Tasks are mixed in a single training set
(caption / VQA / OK-VQA / MCQ / grounding / hallucination QA), which is the
recipe that produced multitask_final_vlm for the 65M model and is now scaled
to a 7B backbone with 4-bit QLoRA.
"""

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA multi-task SFT for Qwen2.5-VL-7B")
    parser.add_argument("--model_path", default="model/qwen25vl-7b-instruct")
    parser.add_argument("--data_paths", nargs="+", required=True,
                        help="parquet files (image_bytes + conversations), mixed into one dataset")
    parser.add_argument("--output_dir", default="out/qwen7b_qlora_multitask")
    parser.add_argument("--max_samples", type=int, default=0, help="cap per-parquet rows for smoke runs")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--max_pixels", type=int, default=768 * 28 * 28,
                        help="Qwen2.5-VL max_pixels (default ~768x768)")
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=0, help="0 = no eval split")
    parser.add_argument("--eval_ratio", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_rows(parquet_paths, limit=0):
    rows = []
    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path, columns=["image_bytes", "conversations"])
        file_rows = []
        for index in range(table.num_rows):
            conversations = json.loads(table.column("conversations")[index].as_py())
            user_turn = next(
                (turn for turn in conversations if turn.get("role") == "user"), None
            )
            assistant_turn = next(
                (turn for turn in conversations if turn.get("role") == "assistant"), None
            )
            if user_turn is None or assistant_turn is None:
                continue
            file_rows.append({
                "image_bytes": table.column("image_bytes")[index].as_py(),
                "user": user_turn.get("content", "").replace("<image>", "").strip(),
                "answer": assistant_turn.get("content", "").strip(),
            })
        if limit:
            file_rows = file_rows[:limit]
        rows.extend(file_rows)
        print(f"loaded {len(rows)} rows from {parquet_path}")
    return rows


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": row["user"]},
                ],
            },
            {"role": "assistant", "content": row["answer"]},
        ]
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full = self.processor(text=full_text, images=image, return_tensors="pt")
        pixel_values = full["pixel_values"]
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.cat(
                [p if p.dim() == 4 else p.squeeze(0) for p in pixel_values], dim=0
            )
        elif pixel_values.dim() == 5:
            pixel_values = pixel_values[0]
        grid_thw = full["image_grid_thw"]
        if isinstance(grid_thw, (list, tuple)):
            grid_thw = torch.stack(grid_thw)
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)
        prompt_text = self.processor.apply_chat_template(
            [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": row["user"]},
                ],
            }],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = self.processor(text=prompt_text, images=image, return_tensors="pt")
        labels = full["input_ids"][0].clone()
        labels[: prompt["input_ids"].shape[1]] = -100
        return {
            "input_ids": full["input_ids"][0],
            "attention_mask": full["attention_mask"][0],
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
        }


def main():
    args = parse_args()
    rows = load_rows(args.data_paths, args.max_samples)
    print(f"total rows: {len(rows)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = MultiTaskDataset(rows, processor)

    def collate_fn(batch):
        max_len = max(item["input_ids"].shape[0] for item in batch)
        pad_id = processor.tokenizer.pad_token_id
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for index, item in enumerate(batch):
            length = item["input_ids"].shape[0]
            input_ids[index, :length] = item["input_ids"]
            attention_mask[index, :length] = item["attention_mask"]
            labels[index, :length] = item["labels"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
            "image_grid_thw": torch.cat([item["image_grid_thw"] for item in batch], dim=0),
        }

    if args.eval_ratio > 0:
        split = int(len(dataset) * (1 - args.eval_ratio))
        train_ds, eval_ds = torch.utils.data.random_split(
            dataset, [split, len(dataset) - split],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_ds, eval_ds = dataset, None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=args.eval_steps,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        seed=args.seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
