"""Official COCOEvalCap evaluation for MiniMind-V checkpoints.

Runs the native VLM on a COCO caption split (default: full val2017),
stores predictions in COCO format, and scores them with pycocoevalcap
(BLEU-1..4, ROUGE-L, CIDEr; METEOR when a Java runtime is available).
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_vlm_benchmark as evb
from model.model_vlm import MiniMindVLM


PROMPT = "<image>\nDescribe this image in one concise sentence."


def parse_args():
    parser = argparse.ArgumentParser(description="Official COCOEvalCap benchmark")
    parser.add_argument("--annotation_file", required=True, help="COCO captions JSON (gt)")
    parser.add_argument("--image_dir", required=True, help="COCO image directory")
    parser.add_argument("--save_dir", default="out")
    parser.add_argument("--weight", default="sft_vlm")
    parser.add_argument("--tokenizer_path", default="model")
    parser.add_argument("--vision_model_path", default="model/siglip2-base-p32-256-ve")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--projector_type", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all records")
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--output_dir", default="results/official_coco")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_generate", action="store_true", help="regenerate predictions even if they exist")
    parser.add_argument("--with_meteor", action="store_true", help="attempt official METEOR (needs Java + Stanford jar)")
    parser.add_argument("--meteor_timeout", type=float, default=90.0, help="seconds to allow METEOR scoring")
    return parser.parse_args()


def build_records(annotation_file, image_dir):
    with open(annotation_file, encoding="utf-8") as file:
        raw = json.load(file)
    names = {item["id"]: item["file_name"] for item in raw["images"]}
    references = {}
    for item in raw["annotations"]:
        references.setdefault(item["image_id"], []).append(item["caption"])
    records = [
        {"image_id": image_id, "image_path": str(Path(image_dir) / names[image_id])}
        for image_id, refs in references.items()
        if image_id in names and (Path(image_dir) / names[image_id]).is_file()
    ]
    return records, references


def generate_predictions(args, records):
    model, tokenizer, processor = evb.load_model(args)
    marker = model.config.image_special_token * model.config.image_token_len
    messages = [{"role": "user", "content": PROMPT.replace("<image>", marker)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(args.device)

    predictions = {}
    start = time.perf_counter()
    for index, record in enumerate(records):
        image = Image.open(record["image_path"]).convert("RGB")
        pixel_values = {
            key: value.to(args.device)
            for key, value in MiniMindVLM.image2tensor(image, processor).items()
        }
        with torch.inference_mode():
            generated = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=pixel_values,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        predictions[record["image_id"]] = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            elapsed = time.perf_counter() - start
            print(f"{index + 1}/{len(records)} ({elapsed / max(index + 1, 1):.3f}s/img)")
    return predictions


def _value(payload):
    if isinstance(payload, dict):
        return float(payload.get("f", sum(payload.values()) / max(len(payload), 1)))
    return float(payload)


def _meteor_worker(gts, res, queue):
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        score, _ = Meteor().compute_score(gts, res)
        queue.put(_value(score))
    except Exception as exc:  # pragma: no cover - runtime dependent
        queue.put(None)


def _meteor_with_timeout(gts, res, timeout):
    """Official METEOR in a daemon subprocess so a hung Java child cannot block the pipeline."""
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_meteor_worker, args=(gts, res, queue), daemon=True)
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5)
        print(f"METEOR timed out after {timeout:.0f}s; skipping")
        return None
    return queue.get() if not queue.empty() else None


def official_scores(references, predictions, meteor_timeout=90.0, with_meteor=False):
    gts = {int(image_id): list(caps) for image_id, caps in references.items() if int(image_id) in predictions}
    res = {int(image_id): [text] for image_id, text in predictions.items() if int(image_id) in references}

    scores = {}
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        bleu, _ = Bleu(4).compute_score(gts, res)
        scores.update({f"Bleu_{n}": float(bleu[n - 1]) for n in (1, 2, 3, 4)})
    except Exception as exc:  # pragma: no cover - depends on install
        print(f"BLEU skipped: {exc}")

    for name, module_path, class_name in (
        ("ROUGE_L", "pycocoevalcap.rouge.rouge", "Rouge"),
        ("CIDEr", "pycocoevalcap.cider.cider", "Cider"),
    ):
        try:
            scorer = getattr(__import__(module_path, fromlist=[class_name]), class_name)()
            score, _ = scorer.compute_score(gts, res)
            scores[name] = _value(score)
        except Exception as exc:  # pragma: no cover
            print(f"{name} skipped: {exc}")

    scores["METEOR"] = _meteor_with_timeout(gts, res, meteor_timeout) if with_meteor else None
    return scores


def main():
    args = parse_args()
    evb.setup_seed(args.seed)
    records, references = build_records(args.annotation_file, args.image_dir)
    if args.max_samples:
        import random
        random.Random(args.seed).shuffle(records)
        records = records[: args.max_samples]
    print(f"evaluating {len(records)} images")

    output_dir = Path(args.output_dir) / args.weight
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / "predictions_coco.json"
    if predictions_file.is_file() and not args.force_generate:
        with open(predictions_file, encoding="utf-8") as file:
            predictions = {int(item["image_id"]): item["caption"] for item in json.load(file)}
        print(f"reusing {len(predictions)} existing predictions from {predictions_file}")
    else:
        predictions = generate_predictions(args, records)
    with open(predictions_file, "w", encoding="utf-8") as output:
        json.dump(
            [{"image_id": image_id, "caption": caption} for image_id, caption in predictions.items()],
            output,
            ensure_ascii=False,
            indent=2,
        )

    scores = official_scores(references, predictions, args.meteor_timeout, args.with_meteor)
    summary = {
        "weight": args.weight,
        "samples": len(predictions),
        "annotation_file": args.annotation_file,
        "official_coco": scores,
        "predictions_file": str(predictions_file),
        "metric_note": "Official COCOEvalCap (pycocoevalcap) scores.",
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
