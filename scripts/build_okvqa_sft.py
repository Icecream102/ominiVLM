"""Build OK-VQA SFT data and deterministic preference pairs (knowledge-type)."""

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
    parser.add_argument("--questions_file", default="dataset/okvqa/OpenEnded_mscoco_train2014_questions.json")
    parser.add_argument("--annotations_file", default="dataset/okvqa/mscoco_train2014_annotations.json")
    parser.add_argument("--image_zip", default="/autodl-pub/data/COCO14/train2014.zip")
    parser.add_argument("--prompt_suffix", default=" Answer in one word or a short phrase.")
    parser.add_argument("--sft_output", default="dataset/okvqa_sft.parquet")
    parser.add_argument("--pairs_output", default="dataset/okvqa_pairs.parquet")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0)
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
        if not counts:
            continue
        chosen = counts.most_common(1)[0][0]
        rejected = None
        for answer, _ in counts.most_common():
            if answer != chosen:
                rejected = answer
                break
        records.append({
            "image_id": q["image_id"],
            "question": q["question"],
            "chosen": chosen,
            "rejected": rejected if rejected else "unknown",
        })
    if args.max_samples:
        import random
        random.Random(args.seed).shuffle(records)
        records = records[: args.max_samples]
    print(f"records: {len(records)}")

    load_image = image_lookup(args.image_zip)
    image_bytes_list, conversations_list = [], []
    pairs_images, pairs_prompts, pairs_chosen, pairs_rejected = [], [], [], []
    for index, record in enumerate(records):
        image = load_image(record["image_id"]).resize((args.image_size, args.image_size))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        image_bytes = buffer.getvalue()
        conversations = [
            {"role": "user", "content": f"<image>\n{record['question']}{args.prompt_suffix}"},
            {"role": "assistant", "content": record["chosen"]},
        ]
        image_bytes_list.append(image_bytes)
        conversations_list.append(json.dumps(conversations, ensure_ascii=False))
        pairs_images.append(image_bytes)
        pairs_prompts.append(record["question"] + args.prompt_suffix)
        pairs_chosen.append(record["chosen"])
        pairs_rejected.append(record["rejected"])
        if (index + 1) % 2000 == 0 or index + 1 == len(records):
            print(f"{index + 1}/{len(records)}")

    Path(args.sft_output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "image_bytes": pa.array(image_bytes_list, type=pa.binary()),
        "conversations": pa.array(conversations_list, type=pa.string()),
    }), args.sft_output, compression="snappy")
    pq.write_table(pa.table({
        "image_bytes": pa.array(pairs_images, type=pa.binary()),
        "prompt": pa.array(pairs_prompts, type=pa.string()),
        "chosen": pa.array(pairs_chosen, type=pa.string()),
        "rejected": pa.array(pairs_rejected, type=pa.string()),
    }), args.pairs_output, compression="snappy")
    print(f"saved {args.sft_output} and {args.pairs_output} ({len(records)} rows)")


if __name__ == "__main__":
    main()
