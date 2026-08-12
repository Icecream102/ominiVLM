"""LoRA SFT of Qwen2.5-VL-3B-Instruct on VQAv2 short-answer data."""

import argparse
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


PROMPT_SUFFIX = " Answer in one word or a short phrase."


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT Qwen2.5-VL on VQA")
    parser.add_argument("--model_path", default="model/qwen25vl-3b-instruct")
    parser.add_argument("--data_path", default="dataset/vqa_sft.parquet")
    parser.add_argument("--output_dir", default="out/qwen_vl_lora")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=512)
    return parser.parse_args()


def load_rows(parquet_path, limit=0):
    table = pq.read_table(parquet_path, columns=["image_bytes", "conversations"])
    rows = []
    for index in range(table.num_rows):
        conversations = json.loads(table.column("conversations")[index].as_py())
        user_content = conversations[0]["content"]
        question = user_content.replace("<image>", "").strip()
        answer = conversations[1]["content"].strip()
        rows.append({
            "image_bytes": table.column("image_bytes")[index].as_py(),
            "question": question,
            "answer": answer,
        })
    return rows[:limit] if limit else rows


class VQADataset(torch.utils.data.Dataset):
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
                    {"type": "text", "text": row["question"] + PROMPT_SUFFIX},
                ],
            },
            {"role": "assistant", "content": row["answer"]},
        ]
        full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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
            [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": row["question"] + PROMPT_SUFFIX},
            ]}],
            tokenize=False, add_generation_prompt=True,
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
    rows = load_rows(args.data_path, args.max_samples)
    print(f"loaded {len(rows)} VQA rows")

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="bfloat16", device_map="cuda"
    )
    dataset = VQADataset(rows, processor)

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

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=20,
        save_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,
        seed=args.seed,
        report_to=[],
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
