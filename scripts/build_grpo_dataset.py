"""Build a deterministic, distributed-across-row-groups GRPO subset."""

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def valid_row(row, min_chars, max_chars):
    conversations = row.get("conversations")
    if isinstance(conversations, str):
        try:
            conversations = json.loads(conversations)
        except json.JSONDecodeError:
            return False
    if not conversations or any(turn.get("tools") or turn.get("functions") for turn in conversations):
        return False
    assistant_index = next(
        (i for i, turn in enumerate(conversations) if turn.get("role") == "assistant"),
        None,
    )
    if assistant_index is None or assistant_index == 0:
        return False
    prompt = " ".join(turn.get("content", "") for turn in conversations[:assistant_index])
    reference = conversations[assistant_index].get("content", "").strip()
    image_bytes = row.get("image_bytes")
    if not image_bytes or "<image>" not in prompt:
        return False
    return min_chars <= len(reference) <= max_chars


def main():
    parser = argparse.ArgumentParser(description="Build MiniMind-V GRPO parquet")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="dataset/grpo_i2t.parquet")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_chars", type=int, default=8)
    parser.add_argument("--max_chars", type=int, default=384)
    args = parser.parse_args()

    source = pq.ParquetFile(args.input)
    group_ids = list(range(source.num_row_groups))
    random.Random(args.seed).shuffle(group_ids)
    selected = []
    rng = random.Random(args.seed + 1)
    for group_id in group_ids:
        rows = source.read_row_group(group_id).to_pylist()
        rng.shuffle(rows)
        selected.extend(
            row for row in rows
            if valid_row(row, args.min_chars, args.max_chars)
        )
        if len(selected) >= args.samples:
            break
    if len(selected) < args.samples:
        raise RuntimeError(f"only found {len(selected)} valid rows, requested {args.samples}")
    selected = selected[:args.samples]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(selected), output, compression="zstd")
    print(f"saved {len(selected)} rows to {output}")


if __name__ == "__main__":
    main()
