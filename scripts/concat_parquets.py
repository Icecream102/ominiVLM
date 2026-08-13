"""Concatenate repository-schema parquet files (image_bytes + conversations)."""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description="Concatenate parquet files")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tables = [pq.read_table(path) for path in args.inputs]
    combined = pa.concat_tables(tables)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(combined, args.output, compression="snappy")
    print(f"concatenated {len(tables)} files -> {args.output} ({combined.num_rows} rows)")


if __name__ == "__main__":
    main()
