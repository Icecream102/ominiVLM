"""Re-score existing official-COCO prediction files with official METEOR.

The official-COCO runs previously produced BLEU/ROUGE-L/CIDEr and skipped
METEOR because the Java runtime was missing or the 90s timeout was too short.
This script recomputes the full official score set (including METEOR) from
saved predictions without touching the GPU, and merges the result into the
existing summary.json files.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_coco_official import official_scores


def load_references(annotation_file):
    with open(annotation_file, encoding="utf-8") as file:
        raw = json.load(file)
    references = {}
    for item in raw["annotations"]:
        references.setdefault(item["image_id"], []).append(item["caption"])
    return references


def main():
    parser = argparse.ArgumentParser(description="Rescore predictions with official METEOR")
    parser.add_argument("--annotation_file", default="dataset/coco2017/annotations/captions_val2017.json")
    parser.add_argument("--results_root", default="results/official_coco")
    parser.add_argument("--weights", nargs="*", default=[], help="checkpoint subdirs; empty = all")
    parser.add_argument("--meteor_timeout", type=float, default=600.0)
    parser.add_argument("--output_root", default=None, help="optional separate output root")
    args = parser.parse_args()

    references = load_references(args.annotation_file)
    root = Path(args.results_root)
    targets = [root / w for w in args.weights] if args.weights else sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "predictions_coco.json").is_file()
    )
    print(f"rescore {len(targets)} checkpoint dirs under {root}")

    for target in targets:
        pred_path = target / "predictions_coco.json"
        if not pred_path.is_file():
            print(f"skip {target}: no predictions_coco.json")
            continue
        with open(pred_path, encoding="utf-8") as file:
            predictions = {int(item["image_id"]): item["caption"] for item in json.load(file)}
        print(f"[{target.name}] scoring {len(predictions)} predictions (METEOR may take minutes)")
        started = time.time()
        scores = official_scores(references, predictions, meteor_timeout=args.meteor_timeout, with_meteor=True)
        print(f"[{target.name}] done in {time.time() - started:.0f}s: {json.dumps(scores)}")

        summary_path = target / "summary.json"
        summary = {}
        if summary_path.is_file():
            with open(summary_path, encoding="utf-8") as file:
                summary = json.load(file)
        summary["official_coco"] = {**summary.get("official_coco", {}), **scores}
        summary["metric_note"] = "Official COCOEvalCap (pycocoevalcap) scores incl. METEOR."
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        print(f"[{target.name}] updated {summary_path}")


if __name__ == "__main__":
    main()
