"""Reward used to decide which generated samples are worth distilling.

``simple_reward`` is reference-based. Swap in a learned reward model here if
you want to curate unlabeled data; keep the same signature.
"""

from __future__ import annotations

from .evaluator import f1_unigram


def simple_reward(prompt: str, prediction: str, reference: str) -> float:
    """Token-overlap F1 between the extracted prediction and the reference."""
    return f1_unigram(prediction, reference)
