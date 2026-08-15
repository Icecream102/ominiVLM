"""Debug DPO log-prob components for the first preference pair."""

import io
import sys
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_qwen_vl_dpo import build_inputs, load_pairs, log_prob


def main():
    model_path = "model/qwen25vl-7b-instruct"
    adapter = "out/qwen7b_knowledge_sft"
    data = "dataset/dpo_v6_combined.parquet"
    pairs = load_pairs(data, 2)
    pair = pairs[0]
    print("pair0 prompt:", pair["prompt"][:80])
    print("pair0 chosen:", repr(pair["chosen"]), "rejected:", repr(pair["rejected"]))

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=401408)
    policy = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda")
    policy = PeftModel.from_pretrained(policy, adapter)
    reference = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda")
    reference = PeftModel.from_pretrained(reference, adapter)
    reference.requires_grad_(False).eval()

    image = Image.open(io.BytesIO(pair["image_bytes"])).convert("RGB")
    chosen = build_inputs(processor, image, pair["prompt"], pair["chosen"])
    rejected = build_inputs(processor, image, pair["prompt"], pair["rejected"])
    print("chosen input len:", chosen["input_ids"].shape, "labels -100 count:",
          int((chosen["labels"] == -100).sum().item()), "answer tokens:",
          int((chosen["labels"] != -100).sum().item()))
    print("rejected input len:", rejected["input_ids"].shape, "answer tokens:",
          int((rejected["labels"] != -100).sum().item()))

    with torch.no_grad():
        pi_chosen = log_prob(policy, chosen, "cuda", grad=False)
        pi_rejected = log_prob(policy, rejected, "cuda", grad=False)
        ref_chosen = log_prob(reference, chosen, "cuda", grad=False)
        ref_rejected = log_prob(reference, rejected, "cuda", grad=False)
    print("pi_chosen:", pi_chosen.tolist())
    print("pi_rejected:", pi_rejected.tolist())
    print("ref_chosen:", ref_chosen.tolist())
    print("ref_rejected:", ref_rejected.tolist())
    ratio = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
    print("log_ratio:", ratio.tolist())
    loss = -torch.nn.functional.logsigmoid(0.1 * ratio).mean().item()
    print("dpo_term:", loss)


if __name__ == "__main__":
    main()
