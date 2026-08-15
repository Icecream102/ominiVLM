"""Build a balanced, quality-filtered DPO preference set (no yes/no single-token dominance)."""

import argparse
import random

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqav2_pairs", default="dataset/vqav2_pairs.parquet")
    parser.add_argument("--okvqa_pairs", default="dataset/okvqa_pairs.parquet")
    parser.add_argument("--vqav2_max", type=int, default=8000)
    parser.add_argument("--output", default="dataset/dpo_v6_combined.parquet")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    vqav2 = pq.read_table(args.vqav2_pairs)
    okvqa = pq.read_table(args.okvqa_pairs)

    chosen = vqav2.column("chosen").to_pylist()
    rejected = vqav2.column("rejected").to_pylist()
    keep = [
        index for index, (ch, rj) in enumerate(zip(chosen, rejected))
        if ch.lower() not in ("yes", "no") and len(ch.split()) >= 2 and len(rj.split()) >= 1
    ]
    rng = random.Random(args.seed)
    keep = rng.sample(keep, min(args.vqav2_max, len(keep)))
    vqav2 = vqav2.take(keep)
    print(f"vqav2 filtered/kept: {len(keep)}")

    schema = pa.schema([
        pa.field("image_bytes", pa.large_binary()),
        pa.field("prompt", pa.large_string()),
        pa.field("chosen", pa.large_string()),
        pa.field("rejected", pa.large_string()),
    ])
    combined = pa.concat_tables([vqav2.cast(schema), okvqa.cast(schema)])
    order = rng.sample(range(combined.num_rows), combined.num_rows)
    pq.write_table(combined.take(order), args.output, compression="snappy")
    print(f"saved {args.output}: {combined.num_rows} rows")


if __name__ == "__main__":
    main()
