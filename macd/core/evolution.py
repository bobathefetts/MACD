"""Strategy genome and the operators that evolve it.

A strategy is a plain JSON-serialisable dict so it can be logged, cached and
diffed. Every gene has a real effect:

* ``generation``  - sampling parameters passed to the model backend
* ``retrieval``   - how many few-shot exemplars to pull from the train split, and how
* ``training``    - LoRA hyperparameters used when this strategy is distilled
* ``prompt_style``- the instruction/answer format
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from typing import Any

PROMPT_STYLES = ("concise", "cot", "bullet", "json")
SELECTORS = ("random", "similar")
RANKS = (4, 8, 16, 32)

GENE_SPACE: dict[str, dict[str, tuple[float, float]]] = {
    "generation": {
        "temperature": (0.0, 1.5),
        "top_p": (0.1, 1.0),
        "max_new_tokens": (16, 256),
    },
    "retrieval": {
        "k": (0, 8),
    },
    "training": {
        "lr": (1e-5, 5e-4),
        "steps": (20, 400),
    },
}

Strategy = dict[str, Any]


def _as_rng(rng: random.Random | int | None) -> random.Random:
    if isinstance(rng, random.Random):
        return rng
    return random.Random(rng)


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def default_strategy() -> Strategy:
    return {
        "generation": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 64},
        "retrieval": {"k": 0, "selector": "random"},
        "training": {"lr": 1e-4, "rank": 8, "steps": 100},
        "prompt_style": "concise",
    }


def normalize_strategy(strategy: Strategy) -> Strategy:
    """Fill missing genes, drop unknown ones, and clamp everything into range."""
    base = default_strategy()
    gen = {**base["generation"], **(strategy.get("generation") or {})}
    ret = {**base["retrieval"], **(strategy.get("retrieval") or {})}
    trn = {**base["training"], **(strategy.get("training") or {})}
    style = strategy.get("prompt_style", base["prompt_style"])
    rank = int(trn["rank"])
    return {
        "generation": {
            "temperature": round(_clamp(float(gen["temperature"]), GENE_SPACE["generation"]["temperature"]), 2),
            "top_p": round(_clamp(float(gen["top_p"]), GENE_SPACE["generation"]["top_p"]), 2),
            "max_new_tokens": int(_clamp(int(gen["max_new_tokens"]), GENE_SPACE["generation"]["max_new_tokens"])),
        },
        "retrieval": {
            "k": int(_clamp(int(ret["k"]), GENE_SPACE["retrieval"]["k"])),
            "selector": ret["selector"] if ret["selector"] in SELECTORS else "random",
        },
        "training": {
            "lr": float(f"{_clamp(float(trn['lr']), GENE_SPACE['training']['lr']):.2e}"),
            "rank": min(RANKS, key=lambda r: abs(r - rank)),
            "steps": int(_clamp(int(trn["steps"]), GENE_SPACE["training"]["steps"])),
        },
        "prompt_style": style if style in PROMPT_STYLES else "concise",
    }


def random_strategy(seed: random.Random | int | None = None) -> Strategy:
    rng = _as_rng(seed)
    lr_lo, lr_hi = GENE_SPACE["training"]["lr"]
    return normalize_strategy(
        {
            "generation": {
                "temperature": rng.uniform(*GENE_SPACE["generation"]["temperature"]),
                "top_p": rng.uniform(*GENE_SPACE["generation"]["top_p"]),
                "max_new_tokens": rng.randint(16, 256),
            },
            "retrieval": {"k": rng.randint(0, 8), "selector": rng.choice(SELECTORS)},
            "training": {
                # learning rates are sampled on a log scale
                "lr": math.exp(rng.uniform(math.log(lr_lo), math.log(lr_hi))),
                "rank": rng.choice(RANKS),
                "steps": rng.randint(20, 400),
            },
            "prompt_style": rng.choice(PROMPT_STYLES),
        }
    )


def mutate(strategy: Strategy, strength: float = 0.2, rng: random.Random | int | None = None) -> Strategy:
    """Return a jittered copy. ``strength`` scales both step size and flip odds."""
    rng = _as_rng(rng)
    child = copy.deepcopy(normalize_strategy(strategy))
    g, r, t = child["generation"], child["retrieval"], child["training"]
    g["temperature"] += rng.uniform(-strength, strength)
    g["top_p"] += rng.uniform(-strength, strength)
    g["max_new_tokens"] += rng.randint(-32, 32)
    r["k"] += rng.randint(-2, 2)
    if rng.random() < strength:
        r["selector"] = rng.choice(SELECTORS)
    t["lr"] *= math.exp(rng.uniform(-strength, strength) * 2)
    if rng.random() < strength:
        t["rank"] = rng.choice(RANKS)
    t["steps"] += rng.randint(-50, 50)
    if rng.random() < strength:
        child["prompt_style"] = rng.choice(PROMPT_STYLES)
    return normalize_strategy(child)


def crossover(a: Strategy, b: Strategy, rng: random.Random | int | None = None) -> Strategy:
    """Uniform crossover at the level of individual genes."""
    rng = _as_rng(rng)
    a, b = normalize_strategy(a), normalize_strategy(b)
    child: Strategy = {}
    for group in ("generation", "retrieval", "training"):
        child[group] = {gene: rng.choice((a, b))[group][gene] for gene in a[group]}
    child["prompt_style"] = rng.choice((a, b))["prompt_style"]
    return normalize_strategy(child)


def strategy_key(strategy: Strategy) -> str:
    """Short stable identifier, used for caching and logs."""
    blob = json.dumps(normalize_strategy(strategy), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
