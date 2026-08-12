"""Concatenate VQA short-answer SFT data with base caption/instruct SFT data.

Mitigates the catastrophic forgetting observed after VQA-only SFT: the mixed
parquet keeps the short-answer VQA format while retaining caption ability.
"""

import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description="Build mixed VQA + caption SFT parquet")
    parser.add_argument("--vqa_parquet", default="dataset/vqa_sft.parquet")
    parser.add_argument("--base_parquet", default="dataset/sft_i2t.parquet")
    parser.add_argument("--vqa_samples", type=int, default=0, help="0 = all VQA rows")
    parser.add_argument("--base_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="dataset/vqa_caption_mix.parquet")
    return parser.parse_args()


def main():
    args = parse_args()
    vqa = pq.read_table(args.vqa_parquet, columns=["image_bytes", "conversations"])
    base = pq.read_table(args.base_parquet, columns=["image_bytes", "conversations"])
    target_schema = pa.schema([
        pa.field("image_bytes", pa.large_binary()),
        pa.field("conversations", pa.large_string()),
    ])
    vqa = vqa.cast(target_schema)
    base = base.cast(target_schema)
    rng = random.Random(args.seed)
    if args.vqa_samples:
        vqa = vqa.take(rng.sample(range(vqa.num_rows), min(args.vqa_samples, vqa.num_rows)))
    indices = rng.sample(range(base.num_rows), min(args.base_samples, base.num_rows))
    base = base.take(indices)
    mixed = pa.concat_tables([vqa, base])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(mixed, args.output, compression="snappy")
    print(f"saved {args.output}: {mixed.num_rows} rows ({vqa.num_rows} VQA + {base.num_rows} base)")


if __name__ == "__main__":
    main()
