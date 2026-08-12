"""Convert AIC-ICC validation annotations and images to MiniMind-V evaluation parquet."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def annotations(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        for key in ("annotations", "data", "images"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError("标注 JSON 应为列表，或含 annotations/data/images 列表")
    return data


def captions(record):
    value = next((record.get(key) for key in ("caption", "captions", "sentence", "sentences") if record.get(key) is not None), None)
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item if isinstance(item, str) else item.get("caption", "") for item in value]
    return []


def main():
    parser = argparse.ArgumentParser(description="将 AIC-ICC 标注转换为 MiniMind-V 评估 Parquet")
    parser.add_argument("--annotations", required=True, help="AIC-ICC validation 标注 JSON")
    parser.add_argument("--image_dir", required=True, help="对应 validation 图片目录")
    parser.add_argument("--output", default="dataset/aic_icc_val.parquet")
    args = parser.parse_args()
    image_dir = Path(args.image_dir)
    lookup = {path.name: path for path in image_dir.rglob("*") if path.is_file()}
    rows, skipped = [], 0
    for record in annotations(args.annotations):
        image_name = next((str(record[key]) for key in ("image_id", "image", "image_name", "file_name") if record.get(key)), None)
        refs = [text.strip() for text in captions(record) if text and text.strip()]
        image_path = lookup.get(Path(image_name).name) if image_name else None
        if not image_path or not refs:
            skipped += 1
            continue
        rows.append({
            "image_id": image_name,
            "image_bytes": image_path.read_bytes(),
            "references": refs,
            "conversations": json.dumps([
                {"role": "user", "content": "<image>\n请描述这张图中的主要物体和场景。"},
                {"role": "assistant", "content": refs[0]},
            ], ensure_ascii=False),
        })
    if not rows:
        raise RuntimeError("没有转换任何样本；请检查 JSON 字段名和图片目录")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)
    print(f"已写入 {len(rows)} 个样本：{output}；跳过 {skipped} 个样本")


if __name__ == "__main__":
    main()
