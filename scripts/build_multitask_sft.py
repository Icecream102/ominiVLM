"""Build a multi-task SFT parquet for the final MiniMind-V training run.

Tasks: COCO2017 caption, VQAv2 train, OK-VQA train and MMBench (en/dev)
multiple-choice. Output follows the sft_i2t.parquet schema used by
trainer/train_sft_vlm.py (image_bytes + conversations).

Data engineering evidence produced alongside the dataset:
  - exact-pair dedup: an (image, instruction) pair is dropped only when the
    same image bytes and the same instruction text repeat (e.g. duplicated
    rows). The same image across tasks or with different questions is kept,
    because multi-task pairs and multi-question-per-image are the desired
    signal.
  - MinHash near-duplicate analysis on normalized instruction text
    (band-based LSH, threshold 0.8) reported as a profile, not a filter,
    because repeated questions across different images are legitimate
    training signals for VQA.
"""

import argparse
import hashlib
import io
import json
import random
import re
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def consensus_answer(annotation):
    counts = Counter()
    for item in annotation.get("answers", []):
        text = item["answer"] if isinstance(item, dict) else str(item)
        counts[normalize_answer(text)] += 1
    return max(counts, key=counts.get) if counts else "unknown"


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


def encode_image(image, size=256, quality=80):
    image = image.resize((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


_ZIP_LOADERS = {}


def get_loader(zip_path):
    if zip_path not in _ZIP_LOADERS:
        _ZIP_LOADERS[zip_path] = image_lookup(zip_path)
    return _ZIP_LOADERS[zip_path]


def load_coco_captions(annotation_file, image_dir_or_zip, samples, seed):
    with open(annotation_file, encoding="utf-8") as file:
        raw = json.load(file)
    names = {item["id"]: item["file_name"] for item in raw["images"]}
    by_image = defaultdict(list)
    for item in raw["annotations"]:
        by_image[item["image_id"]].append(item["caption"])
    image_ids = sorted(by_image)
    rng = random.Random(seed)
    rng.shuffle(image_ids)
    loader = ("dir", str(image_dir_or_zip)) if Path(image_dir_or_zip).is_dir() else ("zip", str(image_dir_or_zip))
    records = []
    for image_id in image_ids[:samples]:
        records.append({
            "image_id": image_id,
            "user": "Describe this image in one concise sentence.",
            "assistant": rng.choice(by_image[image_id]).strip(),
            "task": "caption",
            "image_loader": loader,
        })
    return records


def load_vqa(questions_file, annotations_file, image_zip, samples, seed, task):
    with open(questions_file, encoding="utf-8") as file:
        questions = json.load(file)["questions"]
    with open(annotations_file, encoding="utf-8") as file:
        annotations = json.load(file)["annotations"]
    by_question = {}
    for item in annotations:
        by_question.setdefault(item["question_id"], []).append(item)
    loader = ("dir", str(image_zip)) if Path(image_zip).is_dir() else ("zip", str(image_zip))
    records = [
        {
            "image_id": q["image_id"],
            "user": f"{q['question']} Answer in one word or a short phrase.",
            "assistant": consensus_answer(by_question[q["question_id"]][0]),
            "task": task,
            "image_loader": loader,
        }
        for q in questions
        if q["question_id"] in by_question
    ]
    random.Random(seed).shuffle(records)
    return records[:samples]


def load_mmbench(dataset_dir, samples, seed):
    from datasets import load_from_disk
    data = load_from_disk(dataset_dir)
    indices = list(range(len(data)))
    random.Random(seed).shuffle(indices)
    records = []
    for index in indices[:samples]:
        example = data[int(index)]
        letters = ["A", "B", "C", "D"]
        lines = [example["question"]]
        valid = [(letter, example[letter]) for letter in letters if str(example.get(letter)) != "nan"]
        lines.append("Options:")
        lines += [f"{letter}. {option}" for letter, option in valid]
        lines.append("Answer with the option letter only.")
        records.append({
            "image_id": int(example["index"]),
            "user": "\n".join(lines),
            "assistant": str(example["answer"]),
            "task": "mcq",
            "image_loader": ("pil", example["image"].convert("RGB")),
        })
    return records


def load_image_bytes(record):
    kind, payload = record["image_loader"]
    if kind == "pil":
        image = payload
    elif kind == "dir":
        image_id = record["image_id"]
        candidates = [
            Path(payload) / f"{image_id:012d}.jpg",
            Path(payload) / f"COCO_train2014_{image_id:012d}.jpg",
            Path(payload) / f"COCO_train2017_{image_id:012d}.jpg",
            Path(payload) / f"COCO_val2014_{image_id:012d}.jpg",
            Path(payload) / f"COCO_val2017_{image_id:012d}.jpg",
        ]
        image_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if image_path is None:
            raise FileNotFoundError(f"image {image_id} not found under {payload}")
        image = Image.open(image_path).convert("RGB")
    else:
        image = get_loader(payload)(record["image_id"])
    return encode_image(image)


def normalize_text_for_minhash(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def minhash_profile(records, threshold=0.8, num_perm=128):
    """Band-based LSH near-duplicate analysis on normalized text."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError as exc:
        print(f"minhash profile skipped: {exc}")
        return {"note": "datasketch unavailable"}

    def shingles(text, k=5):
        words = text.split()
        return [" ".join(words[i:i + k]) for i in range(max(len(words) - k + 1, 1))]

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    hashes = {}
    start = time.time()
    for index, record in enumerate(records):
        key = f"record-{index}"
        mh = MinHash(num_perm=num_perm)
        for shingle in set(shingles(normalize_text_for_minhash(record["user"]))):
            mh.update(shingle.encode("utf-8"))
        hashes[index] = mh
        lsh.insert(key, mh)
        if (index + 1) % 50000 == 0:
            print(f"minhash {index + 1}/{len(records)} ({time.time() - start:.0f}s)")

    parent = {}

    def find(node):
        while parent.get(node, node) != node:
            parent[node] = parent.get(parent.get(node, node), parent.get(node, node))
            node = parent[node]
        return node

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for index, mh in hashes.items():
        for other in lsh.query(mh):
            if other.startswith("record-"):
                union(index, int(other.split("-", 1)[1]))
    cluster_sizes = Counter(find(i) for i in range(len(records)))
    near_dupe_items = sum(size - 1 for size in cluster_sizes.values() if size > 1)
    return {
        "threshold": threshold,
        "num_perm": num_perm,
        "samples": len(records),
        "clusters": len(cluster_sizes),
        "near_duplicate_items": near_dupe_items,
        "near_duplicate_rate": near_dupe_items / max(len(records), 1),
        "note": "MinHash(128) + band LSH on normalized user instruction text; reported, not filtered.",
    }


def main():
    parser = argparse.ArgumentParser(description="Build multi-task SFT parquet")
    parser.add_argument("--caption_annotations", default="dataset/coco2017/annotations/captions_train2017.json")
    parser.add_argument("--caption_images", default="/autodl-pub/data/COCO2017/train2017.zip")
    parser.add_argument("--caption_image_dir", default="", help="extracted image directory (faster than zip)")
    parser.add_argument("--caption_samples", type=int, default=120000)
    parser.add_argument("--vqa_questions", default="dataset/vqav2/v2_OpenEnded_mscoco_train2014_questions.json")
    parser.add_argument("--vqa_annotations", default="dataset/vqav2/v2_mscoco_train2014_annotations.json")
    parser.add_argument("--okvqa_questions", default="dataset/okvqa/OpenEnded_mscoco_train2014_questions.json")
    parser.add_argument("--okvqa_annotations", default="dataset/okvqa/mscoco_train2014_annotations.json")
    parser.add_argument("--vqa_samples", type=int, default=100000)
    parser.add_argument("--okvqa_samples", type=int, default=9009)
    parser.add_argument("--mmbench_dir", default="dataset/mmbench_en_dev")
    parser.add_argument("--mmbench_samples", type=int, default=4329)
    parser.add_argument("--image_zip_train2014", default="/autodl-pub/data/COCO14/train2014.zip")
    parser.add_argument("--vqa_image_dir", default="", help="extracted COCO14 train2014 directory")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", default="dataset/multitask_sft.parquet")
    parser.add_argument("--profile_output", default="results/data_profile_multitask.json")
    parser.add_argument("--skip_minhash", action="store_true")
    parser.add_argument("--skip_build", action="store_true", help="only run profiling")
    args = parser.parse_args()

    records = []
    caption_source = args.caption_image_dir or args.caption_images
    vqa_source = args.vqa_image_dir or args.image_zip_train2014
    records += load_coco_captions(args.caption_annotations, caption_source, args.caption_samples, args.seed)
    records += load_vqa(args.vqa_questions, args.vqa_annotations, vqa_source, args.vqa_samples, args.seed, "vqa")
    records += load_vqa(args.okvqa_questions, args.okvqa_annotations, vqa_source, args.okvqa_samples, args.seed, "okvqa")
    records += load_mmbench(args.mmbench_dir, args.mmbench_samples, args.seed)
    random.Random(args.seed).shuffle(records)
    print(f"total records: {len(records)} "
          f"({dict(Counter(r['task'] for r in records))})")

    if args.skip_build:
        print("skip_build: not writing parquet")
    else:
        start = time.time()
        image_hashes = {}
        kept = []
        dropped_duplicate_images = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(load_image_bytes, record): index for index, record in enumerate(records)}
            completed = 0
            for future in as_completed(futures):
                index = futures[future]
                record = records[index]
                try:
                    image_bytes = future.result()
                except Exception as exc:
                    print(f"skip record {index}: {exc}")
                    continue
                digest = hashlib.md5(image_bytes).hexdigest()
                text_digest = hashlib.md5(record["user"].encode()).hexdigest()
                dedup_key = (record["task"], digest, text_digest)
                if dedup_key in image_hashes:
                    dropped_duplicate_images += 1
                    continue
                image_hashes[dedup_key] = index
                conversations = [
                    {"role": "user", "content": f"<image>\n{record['user']}"},
                    {"role": "assistant", "content": record["assistant"]},
                ]
                kept.append({
                    "image_bytes": image_bytes,
                    "conversations": json.dumps(conversations, ensure_ascii=False),
                    "task": record["task"],
                    "image_id": record["image_id"],
                })
                completed += 1
                if completed % 20000 == 0:
                    print(f"encoded {completed}/{len(records)} ({time.time() - start:.0f}s)")
        print(f"image-level dedup: kept {len(kept)}, dropped {dropped_duplicate_images}")

        table = pa.table({
            "image_bytes": pa.array([row["image_bytes"] for row in kept], type=pa.binary()),
            "conversations": pa.array([row["conversations"] for row in kept], type=pa.string()),
        })
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, args.output, compression="snappy")
        print(f"saved {args.output} with {len(kept)} rows")

    profile = {
        "output": args.output,
        "task_counts": dict(Counter(r["task"] for r in records)),
        "image_level_dedup": {
            "total_records": len(records),
            "dropped_duplicate_images": dropped_duplicate_images if not args.skip_build else "skip_build",
            "kept_records": len(kept) if not args.skip_build else "skip_build",
        },
    }
    if not args.skip_minhash:
        profile["minhash_text"] = minhash_profile(records)
    Path(args.profile_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.profile_output, "w", encoding="utf-8") as file:
        json.dump(profile, file, ensure_ascii=False, indent=2)
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print(f"profile saved: {args.profile_output}")


if __name__ == "__main__":
    main()
