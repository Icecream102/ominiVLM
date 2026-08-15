"""Render per-stage sample panels (VQA / OK-VQA / DPO contrast) for the final report."""

import json
import re
import zipfile
from pathlib import Path

import numpy as np
np.Inf = np.inf
np.NaN = np.nan
np.PINF = np.inf
np.NINF = -np.inf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "official_coco_20260812" / "samples"
OUT.mkdir(parents=True, exist_ok=True)
VAL_ZIP = "/autodl-pub/data/COCO14/val2014.zip"


def load_zip(zip_path):
    archive = zipfile.ZipFile(zip_path)
    by_id = {}
    for name in archive.namelist():
        match = re.search(r"(\d{12})\.jpg$", name)
        if match:
            by_id[int(match.group(1))] = name
    return archive, by_id


def load_predictions(path):
    rows = []
    with open(ROOT / path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render(title, panels, filename):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.4))
    if len(panels) == 1:
        axes = [axes]
    for ax, (image, lines) in zip(axes, panels):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title("\n".join(lines), fontsize=8, loc="left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)
    print("saved", OUT / filename)


def main():
    archive, by_id = load_zip(VAL_ZIP)

    def image_of(image_id):
        return Image.open(archive.open(by_id[image_id])).convert("RGB")

    # Stage: VQA (65M multitask vs 7B multitask)
    vqa65 = {d["question_id"]: d for d in load_predictions("results/vqa/multitask_final_vlm/predictions.jsonl")}
    vqa7b = {d["question_id"]: d for d in load_predictions("results/vqa_qwen7b/predictions.jsonl")}
    shared = [q for q in vqa65 if q in vqa7b][:2]
    panels = []
    for question_id in shared:
        image_id = vqa65[question_id]["image_id"]
        q = vqa7b[question_id]["question"][:70]
        panels.append((
            image_of(image_id),
            [f"Q: {q}", f"65M: {vqa65[question_id]['answer'][:40]}", f"7B: {vqa7b[question_id]['answer'][:40]}"],
        ))
    if panels:
        render("VQA stage: 65M multitask vs 7B LoRA", panels, "stage_vqa_65m_vs_7b.png")

    # Stage: OK-VQA (zero-shot vs knowledge SFT vs DPO v6)
    okv = {
        "zero-shot": {d["question_id"]: d for d in load_predictions("results/okvqa/qwen7b_zeroshot/predictions.jsonl")},
        "knowledge SFT": {d["question_id"]: d for d in load_predictions("results/okvqa/qwen7b_knowledge_sft/predictions.jsonl")},
        "DPO v6": {d["question_id"]: d for d in load_predictions("results/okvqa/qwen7b_dpo_v6/predictions.jsonl")},
    }
    shared_q = [q for q in okv["knowledge SFT"] if q in okv["zero-shot"] and q in okv["DPO v6"]][:2]
    panels = []
    for question_id in shared_q:
        row = okv["knowledge SFT"][question_id]
        panels.append((
            image_of(row["image_id"]),
            [f"Q: {row['question'][:70]}",
             f"zero-shot: {okv['zero-shot'][question_id]['answer'][:36]}",
             f"knowledge SFT: {okv['knowledge SFT'][question_id]['answer'][:36]}",
             f"DPO v6: {okv['DPO v6'][question_id]['answer'][:36]}"],
        ))
    if panels:
        render("Knowledge stage: OK-VQA zero-shot vs SFT vs DPO v6", panels, "stage_okvqa_zs_sft_dpo.png")

    # Stage: DPO collapse vs fix (v5 vs v6 on the same OK-VQA question)
    v5 = {d["question_id"]: d for d in load_predictions("results/okvqa/qwen7b_dpo_v5/predictions.jsonl")}
    v6 = {d["question_id"]: d for d in load_predictions("results/okvqa/qwen7b_dpo_v6/predictions.jsonl")}
    shared_q = [q for q in v5 if q in v6][:2]
    panels = []
    for question_id in shared_q:
        row = v6[question_id]
        panels.append((
            image_of(row["image_id"]),
            [f"Q: {row['question'][:70]}",
             f"v5 (collapsed): {v5[question_id]['answer'][:36]}",
             f"v6 (fixed): {v6[question_id]['answer'][:36]}"],
        ))
    if panels:
        render("DPO fix: v5 collapse vs v6 stable", panels, "stage_dpo_v5_vs_v6.png")


if __name__ == "__main__":
    main()
