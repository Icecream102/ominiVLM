"""Create a compact machine-readable summary from a completed experiment."""

import argparse
import json
import re
from pathlib import Path


TRAINING_PATTERN = re.compile(
    r"Training complete, elapsed: (?P<hours>[0-9.]+)h, peak_vram: (?P<vram>[0-9.]+)GB"
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip().startswith("{")]


def mean(records, key):
    return sum(record[key] for record in records) / max(len(records), 1)


def training_summary(path):
    text = Path(path).read_text(encoding="utf-8")
    matches = list(TRAINING_PATTERN.finditer(text))
    if not matches:
        raise ValueError(f"training completion line not found: {path}")
    match = matches[-1]
    return {"elapsed_hours": float(match["hours"]), "peak_vram_gb": float(match["vram"])}


def build_summary(run_dir):
    run_dir = Path(run_dir)
    logs = run_dir / "logs"
    result_root = run_dir / "results/full_pipeline/coco500"
    grpo = read_jsonl(logs / "full_grpo.jsonl")
    window = min(500, len(grpo))
    output = {
        "status": "complete" if (logs / "full_pipeline.done").exists() else "incomplete",
        "training": {
            "pretrain": training_summary(logs / "full_pretrain.log"),
            "sft": training_summary(logs / "full_sft.log"),
            "grpo": {
                "steps": len(grpo),
                "elapsed_hours": grpo[-1]["elapsed_seconds"] / 3600,
                "reward_mean": mean(grpo, "reward_mean"),
                "reward_first_window": mean(grpo[:window], "reward_mean"),
                "reward_last_window": mean(grpo[-window:], "reward_mean"),
                "kl_mean": mean(grpo, "kl"),
                "kl_first_window": mean(grpo[:window], "kl"),
                "kl_last_window": mean(grpo[-window:], "kl"),
            },
        },
        "coco500": {},
    }
    for checkpoint in ("pretrain_full_vlm", "sft_full_vlm", "grpo_full_vlm"):
        with open(result_root / checkpoint / "summary.json", encoding="utf-8") as stream:
            raw = json.load(stream)
        output["coco500"][checkpoint] = {
            "correct": raw["conditions"]["correct"],
            "visual_dependency": raw["visual_dependency"],
        }
    return output


def main():
    parser = argparse.ArgumentParser(description="Summarize a completed MiniMind-V run")
    parser.add_argument("run_dir")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary = build_summary(args.run_dir)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
