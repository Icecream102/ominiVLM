"""Build balanced object-existence (hallucination-targeted) SFT data.

Uses COCO2017 train2017 instance annotations: for each sampled image,
3 positive questions ("Is there a {present object}...") and 3 negative
questions with objects absent from the image (random + popular + adversarial
sampling, mirroring the POPE protocol). This directly targets the yes-bias
hallucination measured by POPE.
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
    parser = argparse.ArgumentParser(description="Build hallucination QA SFT data")
    parser.add_argument("--instances_file", default="dataset/coco2017/annotations/instances_train2017.json")
    parser.add_argument("--image_dir", default="dataset/coco2017/train2017")
    parser.add_argument("--num_images", type=int, default=5000)
    parser.add_argument("--pos_per_image", type=int, default=3)
    parser.add_argument("--neg_per_image", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="dataset/hallucination_sft.parquet")
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
    image_cats = defaultdict(set)
    for ann in data["annotations"]:
        image_cats[ann["image_id"]].add(cat_id_to_name[ann["category_id"]])

    images = sorted(
        image_id for image_id, cats in image_cats.items()
        if len(cats) >= args.pos_per_image
        and (Path(args.image_dir) / f"{image_id:012d}.jpg").is_file()
    )
    rng = random.Random(args.seed)
    rng.shuffle(images)
    selected = images[: args.num_images]
    print(f"selected {len(selected)} images")

    all_categories = sorted(cat_id_to_name.values())
    popularity = Counter(cat_id_to_name[ann["category_id"]] for ann in data["annotations"])
    popular_categories = [name for name, _ in popularity.most_common(10)]
    co_occurrence = defaultdict(Counter)
    for ann in data["annotations"]:
        name = cat_id_to_name[ann["category_id"]]
        for other in image_cats[ann["image_id"]]:
            if other != name:
                co_occurrence[name][other] += 1

    records = []
    for image_id in selected:
        present = set(image_cats[image_id])
        positive = rng.sample(sorted(present), args.pos_per_image)
        absent = sorted(set(all_categories) - present)
        negative = {
            "random": rng.sample(absent, min(args.neg_per_image, len(absent))),
            "popular": rng.sample(
                sorted(set(popular_categories) - present),
                min(args.neg_per_image, len(set(popular_categories) - present)),
            ),
        }
        adversarial_pool = sorted(set().union(*(co_occurrence[cat] for cat in present)) - present)
        negative["adversarial"] = rng.sample(adversarial_pool, min(args.neg_per_image, len(adversarial_pool)))
        for obj in positive:
            records.append({
                "image_id": image_id,
                "path": str(Path(args.image_dir) / f"{image_id:012d}.jpg"),
                "object": obj,
                "answer": "yes",
            })
        for setting, objects in negative.items():
            for obj in objects:
                records.append({
                    "image_id": image_id,
                    "path": str(Path(args.image_dir) / f"{image_id:012d}.jpg"),
                    "object": obj,
                    "answer": "no",
                })
    rng.shuffle(records)
    print(f"records: {len(records)} (yes={sum(r['answer'] == 'yes' for r in records)}, "
          f"no={sum(r['answer'] == 'no' for r in records)})")

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
            digest = hashlib.md5(image_bytes).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            question = f"Is there a {record['object']} in the image? Answer yes or no."
            conversations = [
                {"role": "user", "content": f"<image>\n{question}"},
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
