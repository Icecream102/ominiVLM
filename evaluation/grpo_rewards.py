"""Deterministic reference-based rewards for MiniMind-V GRPO.

The reward intentionally combines complementary exact-token signals and an
anti-repetition term. It does not require a separate learned reward model,
which keeps the post-training experiment reproducible on a single GPU.
"""

import math
from collections import Counter

from evaluation.caption_metrics import meteor_exact, rouge_l, tokenize
from evaluation.caption_metrics import ngrams


def unigram_f1(prediction, reference):
    pred_counts, ref_counts = Counter(prediction), Counter(reference)
    overlap = sum((pred_counts & ref_counts).values())
    if not overlap:
        return 0.0
    precision = overlap / max(len(prediction), 1)
    recall = overlap / max(len(reference), 1)
    return 2 * precision * recall / (precision + recall)


def repetition_ratio(tokens):
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens, tokens[1:]))
    return 1.0 - len(set(bigrams)) / len(bigrams)


DEFAULT_REWARD_WEIGHTS = {
    "unigram_f1": 0.30,
    "rouge_l": 0.30,
    "meteor_exact": 0.30,
    "length_score": 0.10,
    "repetition_penalty": 0.20,
}


def reference_reward(prediction, references, weights=None):
    """Return a bounded, interpretable reward and its components."""
    weights = {**DEFAULT_REWARD_WEIGHTS, **(weights or {})}
    pred = tokenize(prediction)
    refs = [tokenize(reference) for reference in references if reference]
    if not pred or not refs:
        return -1.0, {
            "unigram_f1": 0.0,
            "rouge_l": 0.0,
            "meteor_exact": 0.0,
            "length_score": 0.0,
            "repetition_penalty": 0.0,
        }

    unigram = max(unigram_f1(pred, ref) for ref in refs)
    rouge = rouge_l(pred, refs)
    meteor = meteor_exact(pred, refs)
    closest_length = min((len(ref) for ref in refs), key=lambda n: abs(n - len(pred)))
    length_score = min(len(pred), closest_length) / max(len(pred), closest_length, 1)
    repeat_penalty = repetition_ratio(pred)
    reward = (
        weights["unigram_f1"] * unigram
        + weights["rouge_l"] * rouge
        + weights["meteor_exact"] * meteor
        + weights["length_score"] * length_score
        - weights["repetition_penalty"] * repeat_penalty
    )
    return max(-1.0, min(1.0, reward)), {
        "unigram_f1": unigram,
        "rouge_l": rouge,
        "meteor_exact": meteor,
        "length_score": length_score,
        "repetition_penalty": repeat_penalty,
    }


def build_document_frequency(reference_lists):
    """TF-IDF document frequency over tokenized reference sets, for CIDEr-style rewards."""
    document_frequency = [Counter() for _ in range(4)]
    for refs in reference_lists:
        for n in range(1, 5):
            grams = set().union(*(set(ngrams(ref, n)) for ref in refs))
            document_frequency[n - 1].update(grams)
    return document_frequency, len(reference_lists)


def cider_style_sample(pred, refs, document_frequency, document_count):
    """Per-sample CIDEr-style score on a 0..10 scale (mirrors evaluation.caption_metrics.cider)."""
    n_scores = []
    for n in range(1, 5):
        def weighted(counts):
            return {
                gram: value * math.log(max(1.0, document_count / max(document_frequency[n - 1][gram], 1)))
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
    return 10 * sum(n_scores) / 4


def update_kl_beta(beta, observed_kl, target_kl, rate=0.05, minimum=1e-4, maximum=2.0):
    """Adapt the KL coefficient smoothly toward a target divergence.

    The multiplicative controller is deliberately bounded so a noisy rollout
    cannot collapse the coefficient to zero or make it explode in one step.
    """
    if target_kl <= 0:
        raise ValueError("target_kl must be positive")
    error = max(-1.0, min(1.0, observed_kl / target_kl - 1.0))
    return max(minimum, min(maximum, beta * math.exp(rate * error)))
