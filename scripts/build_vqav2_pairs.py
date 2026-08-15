"""Build deterministic VQAv2 preference pairs: consensus answer vs a wrong answer."""

import argparse
import collections
import io
import json
import re
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_file", default="dataset/vqav2/v2_OpenEnded_mscoco_train2014_questions.json")
    parser.add_argument("--annotations_file", default="dataset/vqav2/v2_mscoco_train2014_annotations.json")
    parser.add_argument("--image_zip", default="/autodl-pub/data/COCO14/train2014.zip")
    parser.add_argument("--prompt_suffix", default=" Answer in one word or a short phrase.")
    parser.add_argument("--output", default="dataset/vqav2_pairs.parquet")
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_lookup(zip_path):
    archive = zipfile.ZipFile(zip_path)
    by_id = {}
    for name in archive.namelist():
        match = re.search(r"(\d{12})\.jpg$", name)
        if match:
            by_id[int(match.group(1))] = name

    def load(image_id):
        member = by_id.get(image_id)
        if member is None:
            raise FileNotFoundError(f"image {image_id} not found")
        return Image.open(archive.open(member)).convert("RGB")

    return load


def main():
    args = parse_args()
    with open(args.questions_file, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    with open(args.annotations_file, encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]
    ann_by_qid = {item["question_id"]: item for item in annotations}

    records = []
    for q in questions:
        ann = ann_by_qid.get(q["question_id"])
        if ann is None:
            continue
        counts = collections.Counter(a["answer"].strip().lower() for a in ann["answers"] if a.get("answer"))
        if len(counts) < 2:
            continue
        chosen = counts.most_common(1)[0][0]
        rejected = next(answer for answer, _ in counts.most_common() if answer != chosen)
        records.append({
            "image_id": q["image_id"],
            "question": q["question"],
            "chosen": chosen,
            "rejected": rejected,
        })
    import random
    random.Random(args.seed).shuffle(records)
    records = records[: args.max_samples]
    print(f"records: {len(records)}")

    load_image = image_lookup(args.image_zip)
    images, prompts, chosen_list, rejected_list = [], [], [], []
    for index, record in enumerate(records):
        image = load_image(record["image_id"]).resize((args.image_size, args.image_size))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        images.append(buffer.getvalue())
        prompts.append(record["question"] + args.prompt_suffix)
        chosen_list.append(record["chosen"])
        rejected_list.append(record["rejected"])
        if (index + 1) % 5000 == 0 or index + 1 == len(records):
            print(f"{index + 1}/{len(records)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "image_bytes": pa.array(images, type=pa.binary()),
        "prompt": pa.array(prompts, type=pa.string()),
        "chosen": pa.array(chosen_list, type=pa.string()),
        "rejected": pa.array(rejected_list, type=pa.string()),
    }), args.output, compression="snappy")
    print(f"saved {args.output} ({len(records)} pairs)")


if __name__ == "__main__":
    main()
