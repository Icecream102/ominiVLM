"""Render side-by-side prediction panels for README/demo purposes.

Reads the per-condition JSONL files produced by eval_vlm_benchmark.py and
writes one PNG per sample: original image + ground truth, then correct,
black, and shuffled predictions.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MiniMind-V COCO predictions")
    parser.add_argument("--predictions_dir", required=True, help="weight output dir from eval_vlm_benchmark")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--annotation_file", required=True)
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--output_dir", default="results/visualizations")
    parser.add_argument("--font_size", type=int, default=18)
    return parser.parse_args()


def load_lines(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def wrap_text(draw, text, width, font):
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= width - 12:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def main():
    args = parse_args()
    root = Path(args.predictions_dir)
    correct = load_lines(root / "predictions_correct.jsonl")
    black = load_lines(root / "predictions_black.jsonl") if (root / "predictions_black.jsonl").is_file() else []
    shuffled = load_lines(root / "predictions_shuffled.jsonl") if (root / "predictions_shuffled.jsonl").is_file() else []

    with open(args.annotation_file, encoding="utf-8") as file:
        raw = json.load(file)
    names = {item["id"]: item["file_name"] for item in raw["images"]}
    image_refs = {}
    for item in raw["annotations"]:
        image_refs.setdefault(item["image_id"], []).append(item["caption"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, args.font_size) if Path(font_path).is_file() else ImageFont.load_default()
    panel_width = 512

    for index, record in enumerate(correct[: args.max_images]):
        image_id = record["target_image_id"]
        image_path = Path(args.image_dir) / names.get(image_id, "")
        if not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB").resize((panel_width, panel_width))
        refs = image_refs.get(image_id, [])
        ref_text = " | ".join(refs)[:220]
        panels = [
            (image, "GT: " + ref_text),
            (image, "correct: " + record["prediction"]),
        ]
        if black:
            panels.append((image, "black: " + black[index]["prediction"]))
        if shuffled:
            panels.append((image, "shuffled: " + shuffled[index]["prediction"]))

        rows, cols = 1, len(panels)
        caption_height = 96
        canvas = Image.new("RGB", (panel_width * cols, panel_width + caption_height), "white")
        draw = ImageDraw.Draw(canvas)
        for col, (img, caption) in enumerate(panels):
            canvas.paste(img, (col * panel_width, 0))
            lines = wrap_text(draw, caption, panel_width, font)[:4]
            y = panel_width + 8
            for line in lines:
                draw.text((col * panel_width + 8, y), line, fill="black", font=font)
                y += args.font_size + 4
        out = output_dir / f"sample_{index:03d}_id{image_id}.png"
        canvas.save(out)
        print(f"saved {out}")

    print(f"visualizations written to {output_dir}")


if __name__ == "__main__":
    main()
