"""Quantitative evaluation for a MiniMind-V SFT checkpoint.

The evaluation parquet must use the same schema as dataset/sft_i2t.parquet:
``image_bytes`` and ``conversations``.  The final assistant turn of each
conversation is treated as the reference answer; all preceding turns form the
generation prompt.  Use a held-out split, not the parquet used for training.
"""

import argparse
import io
import json
import math
import os
import re
from collections import Counter, defaultdict

import jieba
import torch
from datasets import Dataset as HFDataset
from PIL import Image

from model.torch_compat import ensure_torch_transformers_compat
ensure_torch_transformers_compat()
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.trainer_utils import get_model_params, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MiniMind-V SFT with caption metrics")
    parser.add_argument("--data_path", required=True, help="独立测试集 parquet；不要使用训练集")
    parser.add_argument("--load_from", default="model", help="model=原生 pth；其他值=Transformers 模型目录")
    parser.add_argument("--save_dir", default="out", help="原生权重目录")
    parser.add_argument("--weight", default="sft_vlm", help="权重前缀")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, choices=[0, 1], default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=0, help="0 表示评估全部样本")
    parser.add_argument("--results_file", default="eval_results_sft_vlm.jsonl")
    parser.add_argument("--metrics_file", default="eval_metrics_sft_vlm.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from, trust_remote_code=True)
    if args.load_from == "model":
        suffix = "_moe" if args.use_moe else ""
        checkpoint = os.path.join(args.save_dir, f"{args.weight}_{args.hidden_size}{suffix}.pth")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"未找到模型权重：{checkpoint}")
        model = MiniMindVLM(
            VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)),
            vision_model_path="./model/siglip2-base-p32-256-ve",
        )
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict({k: v for k, v in state_dict.items() if "mask" not in k}, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.vision_encoder, model.processor = MiniMindVLM.get_vision_model("./model/siglip2-base-p32-256-ve")
    get_model_params(model, model.config)
    model = model.eval().to(args.device)
    if "cuda" in args.device:
        model = model.half()
    return model, tokenizer, model.processor


def tokens(text):
    """Tokenize Chinese without making a space-separated-caption assumption."""
    text = re.sub(r"\s+", " ", text.strip().lower())
    return [piece for piece in jieba.lcut(text) if piece.strip()]


def ngrams(items, n):
    return Counter(tuple(items[i:i + n]) for i in range(max(0, len(items) - n + 1)))


def corpus_bleu(predictions, references, max_n=4):
    clipped, total = [0] * max_n, [0] * max_n
    pred_length = ref_length = 0
    for pred, refs in zip(predictions, references):
        pred_length += len(pred)
        ref_length += min((len(ref) for ref in refs), key=lambda length: (abs(length - len(pred)), length))
        for n in range(1, max_n + 1):
            pred_counts = ngrams(pred, n)
            max_ref = Counter()
            for ref in refs:
                for gram, count in ngrams(ref, n).items():
                    max_ref[gram] = max(max_ref[gram], count)
            clipped[n - 1] += sum(min(count, max_ref[gram]) for gram, count in pred_counts.items())
            total[n - 1] += sum(pred_counts.values())
    precisions = [(clipped[i] / total[i]) if total[i] else 0.0 for i in range(max_n)]
    bp = 1.0 if pred_length > ref_length else math.exp(1 - ref_length / max(pred_length, 1))
    scores = {}
    for n in range(1, max_n + 1):
        scores[f"BLEU-{n}"] = 0.0 if any(p == 0 for p in precisions[:n]) else bp * math.exp(sum(math.log(p) for p in precisions[:n]) / n)
    return scores


def lcs_length(left, right):
    row = [0] * (len(right) + 1)
    for token in left:
        previous = 0
        for j, other in enumerate(right, 1):
            saved = row[j]
            row[j] = previous + 1 if token == other else max(row[j], row[j - 1])
            previous = saved
    return row[-1]


def rouge_l(pred, refs):
    best = 0.0
    for ref in refs:
        common = lcs_length(pred, ref)
        precision, recall = common / max(len(pred), 1), common / max(len(ref), 1)
        beta_sq = 1.2 ** 2
        score = (1 + beta_sq) * precision * recall / (recall + beta_sq * precision) if precision + recall else 0.0
        best = max(best, score)
    return best


def meteor_exact(pred, refs):
    """METEOR's unigram/chunk formulation, using exact matching for Chinese."""
    best = 0.0
    for ref in refs:
        positions = defaultdict(list)
        for j, word in enumerate(ref):
            positions[word].append(j)
        used, matches, aligned = set(), 0, []
        for word in pred:
            choices = [j for j in positions[word] if j not in used]
            if choices:
                choice = choices[0] if not aligned else min(choices, key=lambda j: (j < aligned[-1], abs(j - aligned[-1])))
                used.add(choice); aligned.append(choice); matches += 1
        if not matches:
            continue
        precision, recall = matches / len(pred), matches / len(ref)
        f_mean = 10 * precision * recall / (recall + 9 * precision)
        chunks = 1 + sum(b != a + 1 for a, b in zip(aligned, aligned[1:]))
        best = max(best, f_mean * (1 - 0.5 * (chunks / matches) ** 3))
    return best


def cider(predictions, references):
    """Corpus CIDEr (n=1..4), with TF-IDF n-grams and the standard scale of 10."""
    document_frequency = [Counter() for _ in range(4)]
    for refs in references:
        for n in range(1, 5):
            document_frequency[n - 1].update(set().union(*(set(ngrams(ref, n)) for ref in refs)))
    count = len(references)
    scores = []
    for pred, refs in zip(predictions, references):
        per_n = []
        for n in range(1, 5):
            pred_counts = ngrams(pred, n)
            def weighted(counts):
                return {
                    gram: value * math.log(
                        max(1.0, count / max(document_frequency[n - 1][gram], 1))
                    )
                    for gram, value in counts.items()
                }
            left = weighted(pred_counts)
            norm_left = math.sqrt(sum(value * value for value in left.values()))
            similarities = []
            for ref in refs:
                right = weighted(ngrams(ref, n))
                norm_right = math.sqrt(sum(value * value for value in right.values()))
                cosine = sum(value * right.get(gram, 0.0) for gram, value in left.items()) / (norm_left * norm_right) if norm_left and norm_right else 0.0
                similarities.append(cosine * math.exp(-((len(pred) - len(ref)) ** 2) / (2 * 6.0 ** 2)))
            per_n.append(sum(similarities) / max(len(similarities), 1))
        scores.append(10 * sum(per_n) / 4)
    return sum(scores) / max(len(scores), 1)


def split_example(conversations):
    if isinstance(conversations, str):
        conversations = json.loads(conversations)
    last_assistant = max((i for i, turn in enumerate(conversations) if turn.get("role") == "assistant"), default=-1)
    if last_assistant < 1:
        raise ValueError("每条样本至少需要一条 assistant 参考答案和其前置 prompt")
    return conversations[:last_assistant], conversations[last_assistant].get("content", "")


def image_inputs(image_bytes, preprocess, device):
    image_bytes = image_bytes if isinstance(image_bytes, list) else [image_bytes]
    inputs = [MiniMindVLM.image2tensor(Image.open(io.BytesIO(blob)).convert("RGB"), preprocess) for blob in image_bytes]
    if hasattr(inputs[0], "keys"):
        return {key: torch.cat([item[key] for item in inputs], dim=0).to(device) for key in inputs[0]}
    return torch.stack(inputs).to(device)


def generate(model, tokenizer, preprocess, history, blobs, args):
    messages = []
    marker = model.config.image_special_token * model.config.image_token_len
    for turn in history:
        content = turn.get("content", "")
        if turn.get("role") != "system":
            content = content.replace("<image>", marker)
        messages.append({"role": turn["role"], "content": content})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(args.device)
    with torch.inference_mode():
        ids = model.generate(**inputs, pixel_values=image_inputs(blobs, preprocess, args.device), max_new_tokens=args.max_new_tokens,
                             do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    args = parse_args()
    setup_seed(args.seed)
    dataset = HFDataset.from_parquet(args.data_path)
    model, tokenizer, preprocess = init_model(args)
    limit = min(len(dataset), args.max_samples) if args.max_samples else len(dataset)
    predictions, references = [], []
    with open(args.results_file, "w", encoding="utf-8") as output:
        for index in range(limit):
            row = dataset[index]
            history, reference = split_example(row["conversations"])
            raw_references = row.get("references") or [reference]
            raw_references = json.loads(raw_references) if isinstance(raw_references, str) and raw_references.startswith("[") else raw_references
            raw_references = [item for item in raw_references if item]
            prediction = generate(model, tokenizer, preprocess, history, row["image_bytes"], args)
            pred_tokens, ref_tokens = tokens(prediction), [tokens(item) for item in raw_references]
            predictions.append(pred_tokens); references.append(ref_tokens)
            output.write(json.dumps({"index": index, "prediction": prediction, "references": raw_references}, ensure_ascii=False) + "\n")
            print(f"[{index + 1}/{limit}] {prediction}")
    metrics = corpus_bleu(predictions, references)
    metrics["METEOR"] = sum(meteor_exact(pred, refs) for pred, refs in zip(predictions, references)) / max(limit, 1)
    metrics["ROUGE-L"] = sum(rouge_l(pred, refs) for pred, refs in zip(predictions, references)) / max(limit, 1)
    metrics["CIDEr"] = cider(predictions, references)
    metrics.update({"samples": limit, "tokenizer": "jieba", "meteor_note": "exact-match METEOR; no synonym/stemming resources"})
    with open(args.metrics_file, "w", encoding="utf-8") as output:
        json.dump(metrics, output, ensure_ascii=False, indent=2)
    print("\nMetrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
