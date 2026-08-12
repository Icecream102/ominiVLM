"""Build a short-answer VQA SFT parquet for MiniMind-V.

Reads VQAv2 train questions/annotations plus COCO2014 train images from the
AutoDL public zip (in-place, no local disk for images), and writes a parquet
matching the sft_i2t.parquet schema used by trainer/train_sft_vlm.py.
"""

import argparse
import io
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq


def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def consensus_answer(annotation):
    counts = Counter(normalize_answer(item["answer"]) for item in annotation["answers"])
    if not counts:
        return "unknown"
    return max(counts, key=counts.get)


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
            raise FileNotFoundError(f"image {image_id} not found in {zip_path}")
        return Image.open(archive.open(member)).convert("RGB")

    return load


def main():
    parser = argparse.ArgumentParser(description="Build VQAv2 short-answer SFT parquet")
    parser.add_argument("--questions_file", required=True)
    parser.add_argument("--annotations_file", required=True)
    parser.add_argument("--image_zip", required=True)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="dataset/vqa_sft.parquet")
    parser.add_argument("--prompt_suffix", default=" Answer in one word or a short phrase.")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    with open(args.questions_file, encoding="utf-8") as file:
        questions = json.load(file)["questions"]
    with open(args.annotations_file, encoding="utf-8") as file:
        annotations = json.load(file)["annotations"]
    answers_by_question = {}
    for item in annotations:
        answers_by_question.setdefault(item["question_id"], []).append(item)

    records = [
        {
            "image_id": q["image_id"],
            "question": q["question"],
            "answer": consensus_answer(answers_by_question[q["question_id"]][0]),
        }
        for q in questions
        if q["question_id"] in answers_by_question
    ]
    random.Random(args.seed).shuffle(records)
    records = records[: args.samples]
    print(f"building {len(records)} VQA SFT samples")

    load_image = image_lookup(args.image_zip)
    image_bytes_list = []
    conversations_list = []
    for index, record in enumerate(records):
        image = load_image(record["image_id"]).resize((args.image_size, args.image_size))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=80)
        conversations = [
            {
                "role": "user",
                "content": f"<image>\n{record['question']}{args.prompt_suffix}",
            },
            {"role": "assistant", "content": record["answer"]},
        ]
        image_bytes_list.append(buffer.getvalue())
        conversations_list.append(json.dumps(conversations, ensure_ascii=False))
        if (index + 1) % 5000 == 0 or index + 1 == len(records):
            print(f"{index + 1}/{len(records)}")

    table = pa.table({
        "image_bytes": pa.array(image_bytes_list, type=pa.binary()),
        "conversations": pa.array(conversations_list, type=pa.string()),
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="snappy")
    print(f"saved {args.output} with {len(records)} rows")


if __name__ == "__main__":
    main()
