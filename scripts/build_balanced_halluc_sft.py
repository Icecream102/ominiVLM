"""Build a balanced yes/no hallucination SFT parquet (equal positive/negative)."""

import argparse
import json
import random

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/hallucination_sft.parquet")
    parser.add_argument("--output", default="dataset/hallucination_balanced.parquet")
    parser.add_argument("--per_class", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def label_of(conversations):
    for turn in conversations:
        if turn.get("role") == "assistant":
            value = turn.get("content", "").strip().lower()
            return "yes" if value.startswith("yes") else "no" if value.startswith("no") else None
    return None


def main():
    args = parse_args()
    table = pq.read_table(args.input, columns=["image_bytes", "conversations"])
    buckets = {"yes": [], "no": []}
    for index in range(table.num_rows):
        conversations = json.loads(table.column("conversations")[index].as_py())
        label = label_of(conversations)
        if label in buckets:
            buckets[label].append(index)
    rng = random.Random(args.seed)
    chosen = rng.sample(buckets["yes"], min(args.per_class, len(buckets["yes"]))) + \
             rng.sample(buckets["no"], min(args.per_class, len(buckets["no"])))
    rng.shuffle(chosen)
    image_bytes = [table.column("image_bytes")[i].as_py() for i in chosen]
    conversations = [table.column("conversations")[i].as_py() for i in chosen]
    out = pa.table({
        "image_bytes": pa.array(image_bytes, type=pa.binary()),
        "conversations": pa.array(conversations, type=pa.string()),
    })
    pq.write_table(out, args.output, compression="snappy")
    print(f"saved {args.output}: {len(chosen)} rows (yes={len(chosen)//2}, no={len(chosen)//2})")


if __name__ == "__main__":
    main()
