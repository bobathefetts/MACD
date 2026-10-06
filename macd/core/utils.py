"""Small shared helpers: seeding, stable hashing, dict merging."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any


def seed_everything(seed: int = 42) -> None:
    """Seed every RNG that is installed. Missing libraries are skipped."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def stable_hash(*parts: Any) -> int:
    """Hash that is identical across processes (unlike the built-in ``hash``)."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def stable_unit(*parts: Any) -> float:
    """Deterministic pseudo-random float in [0, 1) derived from ``parts``."""
    return stable_hash(*parts) / 2**64


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``b`` into a copy of ``a``. Nested dicts are copied."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict):
            base = out.get(k)
            out[k] = deep_merge(base if isinstance(base, dict) else {}, v)
        else:
            out[k] = v
    return out
