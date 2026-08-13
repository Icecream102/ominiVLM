"""Build deterministic hallucination DPO preference pairs.

Reads hallucination_sft.parquet (yes/no QA with ground-truth labels) and
produces DPO pairs: chosen = ground-truth answer, rejected = the opposite.
This teaches the model to refuse non-existent objects instead of hallucinating.
"""

import argparse
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description="Build hallucination DPO pairs")
    parser.add_argument("--input", default="dataset/hallucination_sft.parquet")
    parser.add_argument("--output", default="dataset/hallucination_pairs.parquet")
    parser.add_argument("--max_pairs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_answer(text):
    value = text.strip().lower()
    return "yes" if value.startswith("yes") else "no" if value.startswith("no") else None


def main():
    args = parse_args()
    table = pq.read_table(args.input, columns=["image_bytes", "conversations"])
    pairs = []
    skipped = 0
    for index in range(table.num_rows):
        conversations = json.loads(table.column("conversations")[index].as_py())
        user = next((t.get("content", "") for t in conversations if t.get("role") == "user"), "")
        assistant = next((t.get("content", "") for t in conversations if t.get("role") == "assistant"), "")
        label = normalize_answer(assistant)
        if label is None:
            skipped += 1
            continue
        prompt = user.replace("<image>", "").strip()
        chosen = label
        rejected = "no" if label == "yes" else "yes"
        pairs.append({
            "image_bytes": table.column("image_bytes")[index].as_py(),
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })
    random.Random(args.seed).shuffle(pairs)
    if args.max_pairs:
        yes_pairs = [p for p in pairs if p["chosen"] == "yes"]
        no_pairs = [p for p in pairs if p["chosen"] == "no"]
        per_class = max(1, args.max_pairs // 2)
        pairs = yes_pairs[:per_class] + no_pairs[:per_class]
        random.Random(args.seed).shuffle(pairs)
    yes = sum(1 for p in pairs if p["chosen"] == "yes")
    no = sum(1 for p in pairs if p["chosen"] == "no")
    table_out = pa.table({
        "image_bytes": pa.array([p["image_bytes"] for p in pairs], type=pa.binary()),
        "prompt": pa.array([p["prompt"] for p in pairs], type=pa.string()),
        "chosen": pa.array([p["chosen"] for p in pairs], type=pa.string()),
        "rejected": pa.array([p["rejected"] for p in pairs], type=pa.string()),
    })
    pq.write_table(table_out, args.output, compression="snappy")
    print(f"saved {args.output}: {len(pairs)} pairs (yes={yes}, no={no}, skipped={skipped})")


if __name__ == "__main__":
    main()
