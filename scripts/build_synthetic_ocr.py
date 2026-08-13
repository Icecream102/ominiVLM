"""Build synthetic OCR training/eval data with PIL (no external download).

Renders random words/numbers on noise backgrounds with varied fonts, sizes
and rotations, producing image_bytes + conversations in the repository
schema. This demonstrates synthetic data generation for OCR and provides a
controlled eval set for reading capability. Labeled honestly as synthetic.
"""

import argparse
import io
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


WORDS = [
    "hello", "world", "coffee", "summer", "market", "station", "hotel", "museum",
    "banana", "orange", "purple", "silver", "monday", "friday", "apple", "cloud",
    "river", "mountain", "window", "garden", "travel", "music", "science", "history",
    "book", "table", "chair", "lamp", "phone", "camera", "wallet", "ticket",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build synthetic OCR data")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--eval_samples", type=int, default=300)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="dataset/synthetic_ocr.parquet")
    parser.add_argument("--eval_output", default="dataset/synthetic_ocr_eval.parquet")
    return parser.parse_args()


def font_paths():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    return [path for path in candidates if Path(path).is_file()]


def render_sample(rng, fonts, size):
    count = rng.randint(1, 3)
    text = " ".join(rng.sample(WORDS, count))
    background = (rng.randint(180, 255), rng.randint(180, 255), rng.randint(180, 255))
    foreground = (rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 80))
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(5, 20)):
        draw.line(
            [(rng.randint(0, size), rng.randint(0, size)) for _ in range(2)],
            fill=(rng.randint(120, 200), rng.randint(120, 200), rng.randint(120, 200)),
            width=1,
        )
    font_size = rng.randint(int(size * 0.12), int(size * 0.20))
    font = ImageFont.truetype(rng.choice(fonts), font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2 + rng.randint(-8, 8)
    y = (size - th) // 2 + rng.randint(-8, 8)
    draw.text((x, y), text, font=font, fill=foreground)
    angle = rng.choice([-3, -2, -1, 0, 1, 2, 3])
    if angle:
        image = image.rotate(angle, expand=False, fillcolor=background)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue(), text


def main():
    args = parse_args()
    fonts = font_paths()
    if not fonts:
        raise SystemExit("no system fonts found; install fonts-dejavu-core")
    rng = random.Random(args.seed)

    def build(samples, output_path):
        image_bytes_list = []
        conversations_list = []
        for _ in range(samples):
            image_bytes, text = render_sample(rng, fonts, args.image_size)
            conversations = [
                {"role": "user", "content": "<image>\nRead the text in the image. Output only the text."},
                {"role": "assistant", "content": text},
            ]
            image_bytes_list.append(image_bytes)
            conversations_list.append(json.dumps(conversations, ensure_ascii=False))
        table = pa.table({
            "image_bytes": pa.array(image_bytes_list, type=pa.binary()),
            "conversations": pa.array(conversations_list, type=pa.string()),
        })
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_path, compression="snappy")
        print(f"saved {output_path} with {samples} rows")

    build(args.samples, args.output)
    build(args.eval_samples, args.eval_output)


if __name__ == "__main__":
    main()
