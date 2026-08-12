"""Profile SFT/VQA parquets: language mix, dedup, answer length, image size."""

import argparse
import collections
import hashlib
import io
import json
import random
import re

import pyarrow.parquet as pq
from PIL import Image


def cjk_ratio(text):
    cjk = re.findall(r"[\u3400-\u9fff]", text)
    letters = re.findall(r"[a-z0-9]", text.lower())
    return len(cjk) / max(len(cjk) + len(letters), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", default="dataset/sft_i2t.parquet")
    parser.add_argument("--sample", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/data_profile.json")
    args = parser.parse_args()

    table = pq.read_table(args.parquet, columns=["image_bytes", "conversations"])
    total = table.num_rows
    rng = random.Random(args.seed)
    indices = rng.sample(range(total), min(args.sample, total))
    sampled = table.take(indices)

    languages = collections.Counter()
    answer_lengths = []
    image_bytes_sizes = []
    image_hashes = set()
    roles = collections.Counter()
    questions = 0
    for index in range(sampled.num_rows):
        conversations = json.loads(sampled.column("conversations")[index].as_py())
        for turn in conversations:
            roles[turn.get("role")] += 1
        assistant = next((t.get("content", "") for t in conversations if t.get("role") == "assistant"), "")
        user = next((t.get("content", "") for t in conversations if t.get("role") == "user"), "")
        ratio = cjk_ratio(assistant + " " + user)
        languages["zh" if ratio > 0.5 else ("en" if ratio < 0.2 else "mixed")] += 1
        answer_lengths.append(len(assistant.split()))
        image_bytes = sampled.column("image_bytes")[index].as_py()
        image_bytes_sizes.append(len(image_bytes))
        image_hashes.add(hashlib.md5(image_bytes).hexdigest()[:16])
        if "<image>" in user:
            questions += 1

    sizes = sorted(image_bytes_sizes)
    ans = sorted(answer_lengths)
    profile = {
        "parquet": args.parquet,
        "total_rows": total,
        "sampled_rows": len(indices),
        "language_mix": dict(languages),
        "has_image_placeholder_ratio": round(questions / max(len(indices), 1), 4),
        "unique_image_ratio_sampled": round(len(image_hashes) / max(len(indices), 1), 4),
        "assistant_answer_tokens": {
            "mean": round(sum(answer_lengths) / max(len(answer_lengths), 1), 2),
            "median": ans[len(ans) // 2] if ans else 0,
            "p90": ans[int(len(ans) * 0.9)] if ans else 0,
        },
        "image_bytes": {
            "mean_kb": round(sum(image_bytes_sizes) / max(len(image_bytes_sizes), 1) / 1024, 1),
            "median_kb": round(sizes[len(sizes) // 2] / 1024, 1) if sizes else 0,
            "p90_kb": round(sizes[int(len(sizes) * 0.9)] / 1024, 1) if sizes else 0,
        },
        "role_distribution": dict(roles),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(profile, output, ensure_ascii=False, indent=2)
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
