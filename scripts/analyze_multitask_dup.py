"""Lightweight duplicate-profile for the multi-task SFT parquet.

The full-dataset MinHashLSH pass was too slow (231k records), so this script
reports:
  - exact duplicate rate on (image bytes, instruction text) pairs;
  - unique-image ratio (md5 of image bytes);
  - a sampled MinHash near-duplicate estimate: 15k records compared against
    100 random partners each, Jaccard >= 0.8 counts as near-duplicate.
"""

import argparse
import hashlib
import io
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from datasketch import MinHash


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser(description="Multi-task parquet duplicate profile")
    parser.add_argument("--parquet", default="dataset/multitask_sft.parquet")
    parser.add_argument("--output", default="results/data_profile_multitask.json")
    parser.add_argument("--sample_size", type=int, default=15000)
    parser.add_argument("--compare_per_sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_perm", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    print(f"reading {args.parquet} ...", flush=True)
    table = pq.read_table(args.parquet)
    image_bytes_list = table.column("image_bytes").to_pylist()
    conversations_list = table.column("conversations").to_pylist()
    total = len(image_bytes_list)
    print(f"rows: {total}", flush=True)

    image_hashes = set()
    exact_keys = set()
    texts = []
    duplicate_exact = 0
    unique_images = 0
    for index in range(total):
        image_bytes = image_bytes_list[index]
        if isinstance(image_bytes, list):
            image_bytes = image_bytes[0]
        image_digest = hashlib.md5(bytes(image_bytes)).hexdigest()
        if image_digest not in image_hashes:
            image_hashes.add(image_digest)
            unique_images += 1
        conversations = json.loads(conversations_list[index])
        user_text = next((turn.get("content", "") for turn in conversations if turn.get("role") == "user"), "")
        assistant_text = next((turn.get("content", "") for turn in conversations if turn.get("role") == "assistant"), "")
        text = normalize_text(user_text + " " + assistant_text)
        texts.append(text)
        key = (image_digest, text)
        if key in exact_keys:
            duplicate_exact += 1
        else:
            exact_keys.add(key)

    print(f"exact duplicates: {duplicate_exact} ({duplicate_exact / total:.4f})", flush=True)
    print(f"unique images: {unique_images} ({unique_images / total:.4f})", flush=True)

    def shingles(text, k=5):
        words = text.split()
        return [" ".join(words[i:i + k]) for i in range(max(len(words) - k + 1, 1))]

    rng = random.Random(args.seed)
    sample_indices = rng.sample(range(total), min(args.sample_size, total))
    hashes = {}
    for index in sample_indices:
        mh = MinHash(num_perm=args.num_perm)
        for shingle in set(shingles(texts[index])):
            mh.update(shingle.encode("utf-8"))
        hashes[index] = mh

    near_duplicate_pairs = 0
    comparisons = 0
    for index, mh in hashes.items():
        partners = rng.sample(sample_indices, min(args.compare_per_sample, len(sample_indices)))
        for partner in partners:
            if partner == index:
                continue
            comparisons += 1
            if hashes[partner].jaccard(mh) >= args.threshold:
                near_duplicate_pairs += 1

    near_duplicate_rate = near_duplicate_pairs / max(comparisons, 1)
    print(f"sampled near-duplicate pairs: {near_duplicate_pairs}/{comparisons} "
          f"rate={near_duplicate_rate:.4f}", flush=True)

    profile = {
        "parquet": args.parquet,
        "total_rows": total,
        "exact_duplicate_rows": duplicate_exact,
        "exact_duplicate_rate": duplicate_exact / total,
        "unique_image_ratio": unique_images / total,
        "sampled_minhash_near_duplicate": {
            "sample_size": len(sample_indices),
            "compare_per_sample": args.compare_per_sample,
            "num_perm": args.num_perm,
            "threshold": args.threshold,
            "near_duplicate_pair_rate": near_duplicate_rate,
            "note": "sampled pairwise estimate on normalized instruction text; "
                    "not a filter (repeated questions across images are legitimate VQA signal).",
        },
        "task_counts_expected": {
            "caption": 118287, "vqa": 100000, "okvqa": 9009, "mcq": 4329,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(profile, file, ensure_ascii=False, indent=2)
    print(json.dumps(profile, ensure_ascii=False, indent=2))
    print(f"profile saved: {args.output}")


if __name__ == "__main__":
    main()
