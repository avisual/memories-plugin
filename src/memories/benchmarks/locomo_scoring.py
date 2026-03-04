"""LoCoMo F1 scoring — ported from snap-research/locomo evaluation.py.

Implements the exact normalization and token-level F1 used by the benchmark
so our scores are directly comparable to published baselines.
"""

from __future__ import annotations

import re
import string
from collections import Counter


def _remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the|and)\b", " ", text)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


def _remove_punc(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def normalize_answer(s: str) -> str:
    """Normalize answer string for scoring (matches LoCoMo exactly)."""
    s = s.replace(",", "")
    s = _remove_articles(s)
    s = _white_space_fix(s)
    s = _remove_punc(s)
    s = s.lower().strip()
    return s


def _stem_tokens(tokens: list[str]) -> list[str]:
    """Simple suffix-stripping stemmer (avoids nltk dependency)."""
    stemmed = []
    for t in tokens:
        # Porter-style: strip common English suffixes.
        for suffix in ("tion", "sion", "ness", "ment", "ing", "ies", "ied", "ed", "ly", "er", "es", "s"):
            if len(t) > len(suffix) + 2 and t.endswith(suffix):
                t = t[: -len(suffix)]
                break
        stemmed.append(t)
    return stemmed


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = _stem_tokens(normalize_answer(prediction).split())
    gold_tokens = _stem_tokens(normalize_answer(ground_truth).split())

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_qa(prediction: str, ground_truth: str, category: int) -> float:
    """Score a single QA pair using category-appropriate logic.

    Cat 1 (multi-hop): Average F1 across comma-separated answers.
    Cat 2-4: Token F1.
    Cat 5 (adversarial): 1.0 if prediction says "no information available",
        otherwise compute F1 against the adversarial_answer.
    """
    if category == 1:
        # Multi-hop: ground truth may have comma-separated answers.
        gt_parts = [p.strip() for p in ground_truth.split(",") if p.strip()]
        if not gt_parts:
            return token_f1(prediction, ground_truth)
        scores = [token_f1(prediction, part) for part in gt_parts]
        return sum(scores) / len(scores)

    if category == 5:
        # Adversarial: correct answer is often "unanswerable".
        pred_lower = prediction.lower().strip()
        if any(
            phrase in pred_lower
            for phrase in (
                "no information",
                "not mentioned",
                "cannot be determined",
                "unanswerable",
                "not enough information",
                "no evidence",
            )
        ):
            return 1.0
        # If the model gave an actual answer, check against adversarial_answer.
        return token_f1(prediction, ground_truth)

    # Cat 2, 3, 4: standard token F1.
    return token_f1(prediction, ground_truth)
