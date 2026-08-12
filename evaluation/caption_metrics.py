"""Dependency-light caption metrics used by the local MiniMind-V benchmark.

These implementations are intended for controlled comparisons between
checkpoints in this repository. For paper-to-paper comparisons, use the exact
tokenizer and official metric implementation required by the benchmark.
"""

import math
import re
from collections import Counter, defaultdict


def tokenize(text):
    """Tokenize English words/numbers and individual CJK characters."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[\u3400-\u9fff]", text.lower())


def ngrams(items, n):
    return Counter(tuple(items[i:i + n]) for i in range(max(0, len(items) - n + 1)))


def corpus_bleu(predictions, references, max_n=4):
    clipped, total = [0] * max_n, [0] * max_n
    pred_length = ref_length = 0
    for pred, refs in zip(predictions, references):
        pred_length += len(pred)
        ref_length += min(
            (len(ref) for ref in refs),
            key=lambda length: (abs(length - len(pred)), length),
        )
        for n in range(1, max_n + 1):
            pred_counts = ngrams(pred, n)
            max_ref = Counter()
            for ref in refs:
                for gram, count in ngrams(ref, n).items():
                    max_ref[gram] = max(max_ref[gram], count)
            clipped[n - 1] += sum(
                min(count, max_ref[gram]) for gram, count in pred_counts.items()
            )
            total[n - 1] += sum(pred_counts.values())

    precisions = [clipped[i] / total[i] if total[i] else 0.0 for i in range(max_n)]
    bp = 1.0 if pred_length > ref_length else math.exp(1 - ref_length / max(pred_length, 1))
    scores = {}
    for n in range(1, max_n + 1):
        values = precisions[:n]
        scores[f"BLEU-{n}"] = (
            0.0 if any(value == 0 for value in values)
            else bp * math.exp(sum(math.log(value) for value in values) / n)
        )
    return scores


def lcs_length(left, right):
    row = [0] * (len(right) + 1)
    for token in left:
        previous = 0
        for index, other in enumerate(right, 1):
            saved = row[index]
            row[index] = previous + 1 if token == other else max(row[index], row[index - 1])
            previous = saved
    return row[-1]


def rouge_l(pred, refs):
    best = 0.0
    beta_sq = 1.2 ** 2
    for ref in refs:
        common = lcs_length(pred, ref)
        precision = common / max(len(pred), 1)
        recall = common / max(len(ref), 1)
        score = (
            (1 + beta_sq) * precision * recall / (recall + beta_sq * precision)
            if precision + recall else 0.0
        )
        best = max(best, score)
    return best


def meteor_exact(pred, refs):
    """METEOR unigram/chunk formula using exact token matches only."""
    best = 0.0
    for ref in refs:
        positions = defaultdict(list)
        for index, word in enumerate(ref):
            positions[word].append(index)
        used, aligned = set(), []
        for word in pred:
            choices = [index for index in positions[word] if index not in used]
            if not choices:
                continue
            choice = choices[0] if not aligned else min(
                choices, key=lambda index: (index < aligned[-1], abs(index - aligned[-1]))
            )
            used.add(choice)
            aligned.append(choice)
        matches = len(aligned)
        if not matches:
            continue
        precision, recall = matches / max(len(pred), 1), matches / max(len(ref), 1)
        f_mean = 10 * precision * recall / (recall + 9 * precision)
        chunks = 1 + sum(right != left + 1 for left, right in zip(aligned, aligned[1:]))
        best = max(best, f_mean * (1 - 0.5 * (chunks / matches) ** 3))
    return best


def cider(predictions, references):
    """CIDEr-style TF-IDF n-gram similarity on a 0..10 scale."""
    document_frequency = [Counter() for _ in range(4)]
    for refs in references:
        for n in range(1, 5):
            grams = set().union(*(set(ngrams(ref, n)) for ref in refs))
            document_frequency[n - 1].update(grams)

    document_count = len(references)
    sample_scores = []
    for pred, refs in zip(predictions, references):
        n_scores = []
        for n in range(1, 5):
            def weighted(counts):
                return {
                    gram: value * math.log(
                        max(1.0, document_count / max(document_frequency[n - 1][gram], 1))
                    )
                    for gram, value in counts.items()
                }

            left = weighted(ngrams(pred, n))
            left_norm = math.sqrt(sum(value * value for value in left.values()))
            similarities = []
            for ref in refs:
                right = weighted(ngrams(ref, n))
                right_norm = math.sqrt(sum(value * value for value in right.values()))
                cosine = (
                    sum(value * right.get(gram, 0.0) for gram, value in left.items())
                    / (left_norm * right_norm)
                    if left_norm and right_norm else 0.0
                )
                length_penalty = math.exp(-((len(pred) - len(ref)) ** 2) / (2 * 6.0 ** 2))
                similarities.append(cosine * length_penalty)
            n_scores.append(sum(similarities) / max(len(similarities), 1))
        sample_scores.append(10 * sum(n_scores) / 4)
    return sum(sample_scores) / max(len(sample_scores), 1)


def compute_caption_metrics(prediction_texts, reference_texts):
    predictions = [tokenize(text) for text in prediction_texts]
    references = [[tokenize(text) for text in refs] for refs in reference_texts]
    scores = corpus_bleu(predictions, references)
    count = max(len(predictions), 1)
    scores["METEOR-exact"] = sum(
        meteor_exact(pred, refs) for pred, refs in zip(predictions, references)
    ) / count
    scores["ROUGE-L"] = sum(
        rouge_l(pred, refs) for pred, refs in zip(predictions, references)
    ) / count
    scores["CIDEr"] = cider(predictions, references)
    lengths = [len(pred) for pred in predictions]
    scores["avg_generated_tokens"] = sum(lengths) / count
    scores["empty_rate"] = sum(not pred for pred in predictions) / count
    return scores


def jaccard_distance(left, right):
    left_tokens, right_tokens = set(tokenize(left)), set(tokenize(right))
    union = left_tokens | right_tokens
    return 0.0 if not union else 1 - len(left_tokens & right_tokens) / len(union)
