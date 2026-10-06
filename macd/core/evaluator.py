"""Task metrics. Every evaluator returns a score in [0, 1], higher is better."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any

_PUNCT = str.maketrans("", "", string.punctuation)


def _normalize(s: str) -> str:
    return " ".join(s.lower().translate(_PUNCT).split())


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(ref) else 0.0


def f1_unigram(pred: str, ref: str) -> float:
    p_tokens = _normalize(pred).split()
    r_tokens = _normalize(ref).split()
    if not p_tokens and not r_tokens:
        return 1.0
    if not p_tokens or not r_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for t in r_tokens:
        counts[t] = counts.get(t, 0) + 1
    common = 0
    for t in p_tokens:
        if counts.get(t, 0) > 0:
            common += 1
            counts[t] -= 1
    if common == 0:
        return 0.0
    precision = common / len(p_tokens)
    recall = common / len(r_tokens)
    return 2 * precision * recall / (precision + recall)


def bootstrap_ci(
    scores: list[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``scores``."""
    if not scores:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(scores)
    means = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(n_resamples))
    lo = means[int((alpha / 2) * (n_resamples - 1))]
    hi = means[int((1 - alpha / 2) * (n_resamples - 1))]
    return (lo, hi)


@dataclass
class EvalResult:
    metric: float
    details: dict[str, Any] = field(default_factory=dict)
    per_example: list[float] = field(default_factory=list)


class Evaluator:
    def score_one(self, prediction: str, reference: str) -> float:
        raise NotImplementedError

    def evaluate(self, predictions: list[str], references: list[str]) -> EvalResult:
        if len(predictions) != len(references):
            raise ValueError("predictions and references must be the same length")
        if not predictions:
            return EvalResult(metric=0.0, details={"n": 0})
        scores = [self.score_one(p, r) for p, r in zip(predictions, references, strict=True)]
        metric = sum(scores) / len(scores)
        return EvalResult(metric=metric, details=self._details(predictions, references), per_example=scores)

    def _details(self, predictions: list[str], references: list[str]) -> dict[str, Any]:
        return {"n": len(predictions)}


class QAEvaluator(Evaluator):
    """0.7 * exact match + 0.3 * unigram F1."""

    def score_one(self, prediction: str, reference: str) -> float:
        return 0.7 * exact_match(prediction, reference) + 0.3 * f1_unigram(prediction, reference)

    def _details(self, predictions: list[str], references: list[str]) -> dict[str, Any]:
        n = len(predictions)
        em = sum(exact_match(p, r) for p, r in zip(predictions, references, strict=True)) / n
        f1 = sum(f1_unigram(p, r) for p, r in zip(predictions, references, strict=True)) / n
        return {"n": n, "em": em, "f1": f1}


class SummarizationEvaluator(Evaluator):
    """Unigram F1, a lightweight stand-in for ROUGE-1."""

    def score_one(self, prediction: str, reference: str) -> float:
        return f1_unigram(prediction, reference)

    def _details(self, predictions: list[str], references: list[str]) -> dict[str, Any]:
        n = len(predictions)
        f1 = sum(f1_unigram(p, r) for p, r in zip(predictions, references, strict=True)) / n
        return {"n": n, "rouge1_proxy_f1": f1}
