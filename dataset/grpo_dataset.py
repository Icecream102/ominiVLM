import io
import json

import torch
from datasets import Dataset as HFDataset
from PIL import Image
from torch.utils.data import Dataset

from model.model_vlm import MiniMindVLM


class VLMGRPODataset(Dataset):
    """Single-image prompts paired with reference answers for online GRPO."""

    def __init__(
        self,
        parquet_path,
        tokenizer,
        preprocess,
        max_prompt_length=384,
        image_special_token="<|image_pad|>",
        image_token_len=64,
    ):
        self.dataset = HFDataset.from_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.max_prompt_length = max_prompt_length
        self.image_marker = image_special_token * image_token_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        conversations = row["conversations"]
        if isinstance(conversations, str):
            conversations = json.loads(conversations)
        assistant_index = next(
            (i for i, turn in enumerate(conversations) if turn.get("role") == "assistant"),
            None,
        )
        if assistant_index is None or assistant_index == 0:
            raise ValueError(f"row {index} has no prompt/reference pair")
        prompt_turns = []
        for turn in conversations[:assistant_index]:
            content = turn.get("content", "")
            if turn.get("role") != "system":
                content = content.replace("<image>", self.image_marker)
            prompt_turns.append({"role": turn["role"], "content": content})
        prompt = self.tokenizer.apply_chat_template(
            prompt_turns,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        image_bytes = row["image_bytes"]
        if isinstance(image_bytes, list):
            image_bytes = image_bytes[0]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pixel_values = MiniMindVLM.image2tensor(image, self.preprocess)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "pixel_values": pixel_values,
            "reference": conversations[assistant_index].get("content", ""),
            "row_index": index,
        }

    def references(self):
        """All assistant references (for batch-level TF-IDF reward statistics)."""
        refs = []
        for conversations in self.dataset["conversations"]:
            if isinstance(conversations, str):
                conversations = json.loads(conversations)
            assistant_index = next(
                (i for i, turn in enumerate(conversations) if turn.get("role") == "assistant"),
                None,
            )
            if assistant_index is not None:
                refs.append([conversations[assistant_index].get("content", "")])
        return refs


def grpo_collate_fn(batch):
    if len(batch) != 1:
        raise ValueError("VLM GRPO currently requires --batch_size 1; use group_size for rollouts")
    return batch[0]
