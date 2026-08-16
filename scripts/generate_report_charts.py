"""Generate result charts for the omniVLM README/report."""

import json
from pathlib import Path

import numpy as np
np.Inf = np.inf
np.NaN = np.nan
np.PINF = np.inf
np.NINF = -np.inf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmark_results" / "official_coco_20260812" / "charts"
OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    try:
        with open(ROOT / path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def coco_score(path, key):
    s = load(path)
    return s.get("official_coco", {}).get(key)


def acc(path):
    s = load(path)
    return s.get("accuracy")


def main():
    # 1) Benchmark overview: 65M multitask vs 3B LoRA vs 7B LoRA
    #    one standalone bar chart per metric (VQAv2 / MMBench / COCO CIDEr),
    #    CIDEr shown at raw scale (no x100) with explicit y-axis label.
    models = ["65M multitask", "3B LoRA", "7B LoRA"]
    vqa = [acc("results/vqa/multitask_final_vlm/summary.json"),
           acc("results/vqa_qwen/summary.json"),
           acc("results/vqa_qwen7b/summary.json")]
    mmb = [acc("results/mmbench/multitask_final_vlm_full/summary.json"),
           acc("results/mmbench/qwen3b/summary.json"),
           acc("results/mmbench/qwen7b-qlora/summary.json")]
    cider = [coco_score("results/official_coco/multitask_final_vlm/summary.json", "CIDEr"),
             coco_score("results/official_coco_qwen/summary.json", "CIDEr"),
             coco_score("results/official_coco_qwen7b/summary.json", "CIDEr")]

    palette = ["#4C72B0", "#DD8452", "#55A868"]

    def overview_chart(data, fmt, filename, value_offset):
        values = [v if v else 0 for v in data]
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.bar(models, values, color=palette)
        for i, v in enumerate(values):
            ax.text(i, v + value_offset, f"{v:{fmt}}", ha="center", fontsize=10)
        ax.set_ylim(0, max(values) * 1.18 + value_offset)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=150)
        plt.close(fig)
        print("saved", OUT / filename)

    overview_chart(
        [round(v * 100, 1) if v else 0 for v in vqa],
        ".1f", "benchmark_overview_vqa.png", 0.5,
    )
    overview_chart(
        [round(v * 100, 1) if v else 0 for v in mmb],
        ".1f", "benchmark_overview_mmbench.png", 0.5,
    )
    overview_chart(
        cider, ".3f", "benchmark_overview_cider.png", max(cider) * 0.02,
    )

    # 2) GRPO fix chain (official BLEU-4)
    chain = [
        ("GRPO 原版", coco_score("results/official_coco/grpo_full_vlm/summary.json", "Bleu_4")),
        ("beta=0.10", coco_score("results/official_coco/grpo_beta010_vlm/summary.json", "Bleu_4")),
        ("自适应 KL", coco_score("results/official_coco/grpo_adaptive_vlm/summary.json", "Bleu_4")),
        ("修复版", coco_score("results/official_coco/grpo_fix_vlm/summary.json", "Bleu_4")),
        ("fix2", coco_score("results/official_coco/grpo_fix2_vlm/summary.json", "Bleu_4")),
    ]
    sft4 = coco_score("results/official_coco/sft_full_vlm/summary.json", "Bleu_4")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [c[0] for c in chain]
    values = [c[1] for c in chain]
    ax.plot(labels, values, marker="o", label="GRPO 修复链")
    ax.axhline(sft4, ls="--", color="gray", label=f"SFT 基线 ({sft4:.4f})")
    for i, v in enumerate(values):
        ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_ylabel("Official COCO BLEU-4")
    ax.set_title("GRPO reward-misalignment fix chain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "grpo_fix_chain.png", dpi=150)
    plt.close(fig)

    # 3) Hallucination tradeoff (POPE): rand yes rate vs positive F1
    pope = [
        ("65M 基线", "results/pope/multitask_final_vlm/summary.json"),
        ("65M v1", "results/pope/multitask_halluc_vlm/summary.json"),
        ("65M v2", "results/pope/multitask_halluc_v2_vlm/summary.json"),
        ("7B LoRA", "results/pope/qwen7b-qlora/summary.json"),
        ("7B 幻觉DPO", "results/pope/qwen7b_halluc_dpo/summary.json"),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, path in pope:
        s = load(path)
        st = s.get("settings", {})
        x = st.get("random", {}).get("yes_ratio", 0)
        y = st.get("positive", {}).get("f1", 0)
        ax.scatter(x, y, s=90)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, -6), fontsize=9)
    ax.set_xlabel("POPE random-negative yes rate (lower = less hallucination)")
    ax.set_ylabel("POPE positive F1")
    ax.set_title("Hallucination control tradeoff (data ratio / scale)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "hallucination_tradeoff.png", dpi=150)
    plt.close(fig)

    # 4) 7B three-stage chain MMBench
    stages = ["7B zero-shot", "Cont. pretrain", "+DPO"]
    mmb_chain = [acc("results/mmbench/qwen7b_zeroshot/summary.json"),
                 acc("results/mmbench/qwen7b_contpretrain/summary.json"),
                 acc("results/mmbench/qwen7b_contpretrain_dpo/summary.json")]
    ref = acc("results/mmbench/qwen7b-qlora/summary.json")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(stages, [v * 100 for v in mmb_chain], marker="o", label="本实验链条")
    ax.axhline(ref * 100, ls="--", color="gray", label=f"multitask SFT 参考 ({ref*100:.1f}%)")
    for i, v in enumerate(mmb_chain):
        ax.annotate(f"{v*100:.2f}%", (i, v * 100), textcoords="offset points", xytext=(0, 8), ha="center")
    ax.set_ylabel("MMBench en/dev acc (%)")
    ax.set_title("7B continual-pretrain -> DPO chain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "qwen7b_chain_mmbench.png", dpi=150)
    plt.close(fig)

    # 5) Mix ratio VQA
    ratios = ["1:3", "1:1", "3:1"]
    mix = [acc("results/vqa_ratio/vqa_mix_1to3/summary.json"),
           0.209,  # vqa_sft_mix (1:1) official subset
           acc("results/vqa_ratio/vqa_mix_3to1/summary.json")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(ratios, [v * 100 for v in mix], color="#DD8452")
    for i, v in enumerate(mix):
        ax.text(i, v * 100 + 0.4, f"{v*100:.1f}%", ha="center")
    ax.set_ylabel("VQAv2 acc (%)")
    ax.set_title("VQA:caption data-mixing ratio ablation")
    fig.tight_layout()
    fig.savefig(OUT / "mix_ratio_vqa.png", dpi=150)
    plt.close(fig)

    print(f"charts saved to {OUT}")


if __name__ == "__main__":
    main()
