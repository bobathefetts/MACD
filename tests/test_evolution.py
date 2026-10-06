import random

from macd.core.evolution import (
    GENE_SPACE,
    PROMPT_STYLES,
    RANKS,
    crossover,
    mutate,
    normalize_strategy,
    random_strategy,
    strategy_key,
)


def _in_range(s):
    g, r, t = s["generation"], s["retrieval"], s["training"]
    assert GENE_SPACE["generation"]["temperature"][0] <= g["temperature"] <= GENE_SPACE["generation"]["temperature"][1]
    assert 0.1 <= g["top_p"] <= 1.0
    assert 16 <= g["max_new_tokens"] <= 256
    assert 0 <= r["k"] <= 8
    assert 1e-5 <= t["lr"] <= 5e-4
    assert t["rank"] in RANKS
    assert 20 <= t["steps"] <= 400
    assert s["prompt_style"] in PROMPT_STYLES


def test_random_strategy_is_seeded_and_valid():
    assert random_strategy(3) == random_strategy(3)
    assert random_strategy(3) != random_strategy(4)
    for seed in range(50):
        _in_range(random_strategy(seed))


def test_mutation_stays_in_range_and_does_not_touch_parent():
    rng = random.Random(0)
    parent = random_strategy(1)
    snapshot = strategy_key(parent)
    current = parent
    for _ in range(200):
        current = mutate(current, strength=1.0, rng=rng)
        _in_range(current)
    assert strategy_key(parent) == snapshot


def test_mutation_is_reproducible():
    parent = random_strategy(1)
    assert mutate(parent, rng=random.Random(5)) == mutate(parent, rng=random.Random(5))


def test_crossover_takes_genes_from_parents():
    a, b = random_strategy(1), random_strategy(2)
    child = crossover(a, b, random.Random(0))
    for group in ("generation", "retrieval", "training"):
        for gene, value in child[group].items():
            assert value in (a[group][gene], b[group][gene])


def test_normalize_handles_legacy_and_partial_strategies():
    legacy = {"retrieval": {"k": 99, "chunk_size": 500}, "prompt_style": "nonsense", "training": {"rank": 20}}
    s = normalize_strategy(legacy)
    _in_range(s)
    assert "chunk_size" not in s["retrieval"]
    assert s["retrieval"]["k"] == 8
