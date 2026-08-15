"""Official COCOEvalCap evaluation for Qwen2.5-VL + LoRA adapter."""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="COCO caption eval for Qwen2.5-VL LoRA")
    parser.add_argument("--model_path", default="model/qwen25vl-3b-instruct")
    parser.add_argument("--adapter_path", default="out/qwen_vl_lora")
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--output_dir", default="results/official_coco_qwen")
    parser.add_argument("--tag", default="qwen", help="output subdirectory")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.annotation_file, encoding="utf-8") as file:
        raw = json.load(file)
    names = {item["id"]: item["file_name"] for item in raw["images"]}
    references = defaultdict(list)
    for item in raw["annotations"]:
        references[item["image_id"]].append(item["caption"])
    records = [
        {"image_id": image_id, "image_path": str(Path(args.image_dir) / names[image_id])}
        for image_id in references
        if image_id in names and (Path(args.image_dir) / names[image_id]).is_file()
    ]
    if args.max_samples:
        import random
        random.Random(args.seed).shuffle(records)
        records = records[: args.max_samples]
    print(f"evaluating {len(records)} images")

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="bfloat16", device_map="cuda"
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    predictions = {}
    start = time.perf_counter()
    for index, record in enumerate(records):
        image = Image.open(record["image_path"]).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one concise sentence."},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        caption = processor.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        predictions[record["image_id"]] = caption
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            elapsed = time.perf_counter() - start
            print(f"{index + 1}/{len(records)} ({elapsed / max(index + 1, 1):.3f}s/img)")

    output_dir = Path(args.output_dir) / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "predictions_coco.json", "w", encoding="utf-8") as output:
        json.dump([{"image_id": k, "caption": v} for k, v in predictions.items()], output, ensure_ascii=False, indent=2)

    gts = {int(k): list(v) for k, v in references.items() if int(k) in predictions}
    res = {int(k): [v] for k, v in predictions.items() if int(k) in references}
    scores = {}
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        bleu, _ = Bleu(4).compute_score(gts, res)
        scores.update({f"Bleu_{n}": float(bleu[n - 1]) for n in (1, 2, 3, 4)})
    except Exception as exc:
        print(f"BLEU skipped: {exc}")
    for name, module, cls in (
        ("ROUGE_L", "pycocoevalcap.rouge.rouge", "Rouge"),
        ("CIDEr", "pycocoevalcap.cider.cider", "Cider"),
    ):
        try:
            scorer = getattr(__import__(module, fromlist=[cls]), cls)()
            score, _ = scorer.compute_score(gts, res)
            value = score.get("f") if isinstance(score, dict) else score
            scores[name] = float(value)
        except Exception as exc:
            print(f"{name} skipped: {exc}")
    scores["METEOR"] = None
    summary = {
        "weight": "qwen2.5-vl-3b-lora",
        "samples": len(predictions),
        "official_coco": scores,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
