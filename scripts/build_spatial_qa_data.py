"""Build spatial/counting QA SFT data from COCO2017 train2017 instances.

Self-contained visual-reasoning data (no external downloads):
  - counting:   "How many {object}(s) are in the image?" -> integer
  - spatial:    "Where is the {object} in the image?" -> left/right/top/bottom/center
  - existence:  "Is there a {object} in the image?" -> yes/no
Uses object bounding boxes from instances_train2017.json and the already
extracted train2017 images.
"""

import argparse
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Build spatial/counting QA data")
    parser.add_argument("--instances_file", default="dataset/coco2017/annotations/instances_train2017.json")
    parser.add_argument("--image_dir", default="dataset/coco2017/train2017")
    parser.add_argument("--num_images", type=int, default=8000)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="dataset/spatial_qa.parquet")
    return parser.parse_args()


def encode_image(path, size):
    image = Image.open(path).convert("RGB").resize((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return buffer.getvalue()


def main():
    args = parse_args()
    with open(args.instances_file, encoding="utf-8") as file:
        data = json.load(file)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
    image_anns = defaultdict(list)
    for ann in data["annotations"]:
        image_anns[ann["image_id"]].append(ann)

    images = sorted(
        image_id for image_id, anns in image_anns.items()
        if len(anns) >= 2 and (Path(args.image_dir) / f"{image_id:012d}.jpg").is_file()
    )
    rng = random.Random(args.seed)
    rng.shuffle(images)
    selected = images[: args.num_images]
    print(f"selected {len(selected)} images")

    records = []
    for image_id in selected:
        anns = image_anns[image_id]
        by_cat = defaultdict(list)
        for ann in anns:
            by_cat[cat_id_to_name[ann["category_id"]]].append(ann)
        for cat_name, cat_anns in list(by_cat.items()):
            if len(cat_anns) == 1:
                ann = cat_anns[0]
                x, y, w, h = ann["bbox"]
                cx, cy = x + w / 2, y + h / 2
                position = (
                    "center" if 0.35 <= cx / 640 <= 0.65 and 0.35 <= cy / 480 <= 0.65
                    else ("left" if cx / 640 < 0.5 else "right")
                    if abs(cy / 480 - 0.5) <= abs(cx / 640 - 0.5)
                    else ("top" if cy / 480 < 0.5 else "bottom")
                )
                records.append({
                    "image_id": image_id,
                    "path": str(Path(args.image_dir) / f"{image_id:012d}.jpg"),
                    "question": f"Where is the {cat_name} in the image? Answer with a single word: left, right, top, bottom or center.",
                    "answer": position,
                })
        for cat_name, cat_anns in by_cat.items():
            count = len(cat_anns)
            records.append({
                "image_id": image_id,
                "path": str(Path(args.image_dir) / f"{image_id:012d}.jpg"),
                "question": f"How many {cat_name}s are in the image? Answer with an integer only.",
                "answer": str(count),
            })
        present = set(by_cat.keys())
        absent = sorted(set(cat_id_to_name.values()) - present)
        for _ in range(2):
            if not absent:
                break
            obj = rng.choice(absent)
            records.append({
                "image_id": image_id,
                "path": str(Path(args.image_dir) / f"{image_id:012d}.jpg"),
                "question": f"Is there a {obj} in the image? Answer yes or no.",
                "answer": "no",
            })
    rng.shuffle(records)
    print(f"records: {len(records)}")

    image_bytes_list = []
    conversations_list = []
    seen = set()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(encode_image, record["path"], args.image_size): index
                   for index, record in enumerate(records)}
        for future in as_completed(futures):
            index = futures[future]
            record = records[index]
            image_bytes = future.result()
            digest = hashlib.md5(image_bytes + record["question"].encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            conversations = [
                {"role": "user", "content": f"<image>\n{record['question']}"},
                {"role": "assistant", "content": record["answer"]},
            ]
            image_bytes_list.append(image_bytes)
            conversations_list.append(json.dumps(conversations, ensure_ascii=False))

    table = pa.table({
        "image_bytes": pa.array(image_bytes_list, type=pa.binary()),
        "conversations": pa.array(conversations_list, type=pa.string()),
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="snappy")
    print(f"saved {args.output} with {len(image_bytes_list)} rows")


if __name__ == "__main__":
    main()
