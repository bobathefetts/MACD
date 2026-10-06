"""Few-shot exemplar selection from the train split."""

from __future__ import annotations

import random

from ..adapters.base import Example
from .utils import stable_hash


def _tokens(text: str) -> set[str]:
    return set(text.lower().replace("?", " ").replace(".", " ").replace(",", " ").split())


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def select_exemplars(
    example: Example, pool: list[Example], k: int, selector: str = "random", seed: int = 0
) -> list[Example]:
    """Pick ``k`` exemplars for ``example``; never returns the example itself."""
    candidates = [p for p in pool if p.prompt != example.prompt]
    if k <= 0 or not candidates:
        return []
    k = min(k, len(candidates))
    if selector == "similar":
        ranked = sorted(candidates, key=lambda p: (-jaccard(p.prompt, example.prompt), p.prompt))
        # most similar exemplar goes last, closest to the question
        return list(reversed(ranked[:k]))
    rng = random.Random(stable_hash(seed, example.prompt))
    return rng.sample(candidates, k)
