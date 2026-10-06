"""MetaController: generate -> evaluate -> select -> evolve -> distill.

Data discipline:

* ``train``  - few-shot exemplar pool and the only source of distillation data
* ``dev``    - used to select strategies and to accept or reject an adapter
* ``test``   - never used for any decision; reported with a confidence interval
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from ..adapters import Example, get_adapter
from ..backends import Backend, GenerationRequest, TrainingNotSupported, build_backend
from ..config import coerce_model_config, coerce_task_config
from .evaluator import EvalResult, bootstrap_ci
from .evolution import Strategy, crossover, default_strategy, mutate, normalize_strategy, random_strategy, strategy_key
from .retrieval import select_exemplars
from .reward import simple_reward
from .tracking import RunLogger
from .trainer import DistillationStore, Trainer
from .utils import stable_hash


class MetaController:
    def __init__(
        self,
        task_cfg: Any,
        model_cfg: Any,
        seed: int = 42,
        run_dir: str | Path | None = None,
        backend: Backend | None = None,
    ) -> None:
        self.task_cfg = coerce_task_config(task_cfg)
        self.model_cfg = coerce_model_config(model_cfg)
        self.seed = seed
        self.rng = random.Random(seed)
        self.logger = RunLogger(run_dir)
        self.logger.save_config(
            {"task": self.task_cfg.model_dump(), "model": self.model_cfg.model_dump(), "seed": seed}
        )

        self.adapter = get_adapter(self.task_cfg.task.type)
        self.backend = backend or build_backend(self.model_cfg.model.backend, self.model_cfg.model.params)
        adapters_dir = Path(run_dir) / "adapters" if run_dir else Path("outputs") / "adapters"
        self.trainer = Trainer(self.backend, out_dir=adapters_dir)
        self.store = DistillationStore()

        data = self.task_cfg.task.data
        self.splits: dict[str, list[Example]] = {"dev": self.adapter.load(data.dev_path)}
        self.splits["train"] = self.adapter.load(data.train_path) if data.train_path else []
        self.splits["test"] = self.adapter.load(data.test_path) if data.test_path else []
        self._check_leakage()

        self.best_strategy: Strategy = default_strategy()
        self.best_validation: float = 0.0
        self.population: list[Strategy] = []
        self.generation = 0
        self._cache: dict[tuple[str, str | None, str], EvalResult] = {}

    # ------------------------------------------------------------------ data
    def _check_leakage(self) -> None:
        seen: dict[str, str] = {}
        for split, items in self.splits.items():
            for ex in items:
                other = seen.setdefault(ex.prompt, split)
                if other != split:
                    raise ValueError(
                        f"Data leakage: prompt {ex.prompt!r} appears in both the {other} and {split} splits"
                    )

    # ------------------------------------------------------------ generation
    def _requests(self, examples: list[Example], strategy: Strategy) -> list[GenerationRequest]:
        gen, ret, style = strategy["generation"], strategy["retrieval"], strategy["prompt_style"]
        requests = []
        for ex in examples:
            shots = select_exemplars(ex, self.splits["train"], ret["k"], ret["selector"], seed=self.seed)
            requests.append(
                GenerationRequest(
                    messages=self.adapter.build_messages(ex, style, shots),
                    temperature=gen["temperature"],
                    top_p=gen["top_p"],
                    max_new_tokens=gen["max_new_tokens"],
                    seed=stable_hash(self.seed, ex.prompt) % (2**31),
                    hints={"style": style, "task": self.adapter.task_type},
                )
            )
        return requests

    def predict(self, examples: list[Example], strategy: Strategy) -> list[str]:
        """Generate and extract bare answers for ``examples`` under ``strategy``."""
        strategy = normalize_strategy(strategy)
        raw = self.backend.generate(self._requests(examples, strategy))
        return [self.adapter.extract(r, strategy["prompt_style"]) for r in raw]

    # ------------------------------------------------------------ evaluation
    def evaluate(self, strategy: Strategy | None = None, split: str = "dev") -> EvalResult:
        """Score a strategy on a split. Results are cached per (strategy, adapter, split)."""
        strategy = normalize_strategy(strategy or self.best_strategy)
        if not self.splits.get(split):
            raise ValueError(f"No data for split '{split}'. Set task.data.{split}_path in the task config.")
        key = (strategy_key(strategy), self.backend.active_adapter, split)
        if key not in self._cache:
            examples = self.splits[split]
            preds = self.predict(examples, strategy)
            self._cache[key] = self.adapter.evaluator.evaluate(preds, [ex.answer for ex in examples])
        return self._cache[key]

    def evaluate_once(self, strategy: Strategy | None = None, split: str = "dev") -> float:
        return self.evaluate(strategy, split).metric

    def test_report(self, strategy: Strategy | None = None) -> dict[str, Any]:
        """Held-out score with a 95% bootstrap confidence interval."""
        res = self.evaluate(strategy, split="test")
        lo, hi = bootstrap_ci(res.per_example, seed=self.seed)
        return {"metric": res.metric, "ci95": [lo, hi], **res.details}

    # ------------------------------------------------------------- evolution
    def evolve(
        self, population: int | None = None, top_k: int | None = None, generations: int | None = None
    ) -> dict[str, Any]:
        """Run the evolutionary search on the dev split.

        The population persists between calls, so offspring bred at the end of
        one call are the first thing scored in the next.
        """
        search = self.task_cfg.search
        population = population or search.population
        top_k = max(1, min(top_k or search.top_k, population))
        generations = generations or search.generations

        if not self.population:
            self.population = [self.best_strategy]
        self.population = self._dedupe(self.population)[:population]
        while len(self.population) < population:
            self.population.append(random_strategy(self.rng))

        scored: list[tuple[float, Strategy]] = []
        for _ in range(generations):
            self.generation += 1
            scored = sorted(
                ((self.evaluate(s, "dev").metric, s) for s in self.population),
                key=lambda pair: (-pair[0], strategy_key(pair[1])),
            )
            elites = [s for _, s in scored[:top_k]]
            children: list[Strategy] = []
            attempts = 0
            while len(elites) + len(children) < population:
                attempts += 1
                parent = self._tournament(scored)
                if self.rng.random() < search.crossover_rate:
                    parent = crossover(parent, self._tournament(scored), self.rng)
                child = mutate(parent, strength=search.mutation_strength, rng=self.rng)
                known = {strategy_key(s) for s in elites + children}
                if strategy_key(child) in known and attempts < population * 10:
                    continue  # skip clones while we can still find something new
                children.append(child)
            self.population = elites + children
            self.logger.log(
                "generation",
                generation=self.generation,
                best=scored[0][0],
                mean=sum(m for m, _ in scored) / len(scored),
                best_strategy=scored[0][1],
            )

        best_metric, best_strategy = scored[0]
        self.best_strategy = best_strategy
        return {
            "best_metric": best_metric,
            "best_strategy": best_strategy,
            "elites": [s for _, s in scored[:top_k]],
            "population": list(self.population),
            "scores": [m for m, _ in scored],
        }

    def _tournament(self, scored: list[tuple[float, Strategy]], size: int = 3) -> Strategy:
        contenders = [self.rng.choice(scored) for _ in range(min(size, len(scored)))]
        return max(contenders, key=lambda pair: pair[0])[1]

    @staticmethod
    def _dedupe(strategies: list[Strategy]) -> list[Strategy]:
        seen: set[str] = set()
        out = []
        for s in strategies:
            key = strategy_key(s)
            if key not in seen:
                seen.add(key)
                out.append(normalize_strategy(s))
        return out

    # ---------------------------------------------------------- distillation
    def distill_if_improved(self, best_validation: float, threshold: float | None = None) -> dict[str, Any]:
        """Distill the best strategy into an adapter, and keep it only if dev does not regress."""
        cfg = self.task_cfg.training
        threshold = cfg.min_improvement if threshold is None else threshold
        if best_validation <= self.best_validation + threshold:
            return {"distilled": False, "reason": "no dev improvement"}
        self.best_validation = best_validation
        if not self.backend.supports_training:
            return {"distilled": False, "reason": "backend cannot train"}
        if not self.splits["train"]:
            return {"distilled": False, "reason": "no train split"}

        # 1. curate: keep train samples the best strategy gets right
        strategy = self.best_strategy
        train = self.splits["train"]
        added = 0
        for ex, pred in zip(train, self.predict(train, strategy), strict=True):
            reward = simple_reward(ex.prompt, pred, ex.answer)
            added += self.store.add(ex.prompt, pred, ex.answer, reward=reward, threshold=cfg.distill_threshold)
        if len(self.store) < cfg.min_examples:
            return {"distilled": False, "reason": "too few curated examples", "curated": len(self.store)}

        # 2. train: teach the plain prompt to give the answer the full strategy found
        examples = [
            {
                "prompt": row["prompt"],
                "target": row["prediction"],
                "messages": self.adapter.build_messages(Example(row["prompt"], row["reference"]), "concise"),
            }
            for row in self.store.curated
        ]
        previous = self.backend.active_adapter
        try:
            info = self.trainer.distill(examples, hyper=strategy["training"], seed=self.seed)
            self.backend.activate(info["adapter"])
        except TrainingNotSupported as exc:
            self.backend.activate(previous)
            return {"distilled": False, "reason": str(exc), "curated": len(self.store)}

        # 3. gate: roll back if the adapter makes dev worse
        after = self.evaluate(strategy, "dev").metric
        accepted = after + cfg.regression_tolerance >= best_validation
        if accepted:
            self.best_validation = max(best_validation, after)
        else:
            self.backend.activate(previous)
        result = {
            "distilled": accepted,
            "reason": "accepted" if accepted else "adapter regressed on dev; rolled back",
            "curated": len(self.store),
            "newly_curated": added,
            "dev_before": best_validation,
            "dev_after": after,
            "info": info,
        }
        self.logger.log("distill", **result)
        return result

    # ----------------------------------------------------------------- cycle
    def run_cycle(
        self, population: int | None = None, top_k: int | None = None, generations: int | None = None
    ) -> dict[str, Any]:
        """One full MACD cycle: evolve on dev, distill, then report."""
        evo = self.evolve(population=population, top_k=top_k, generations=generations)
        distill = self.distill_if_improved(best_validation=evo["best_metric"])
        record: dict[str, Any] = {
            "val": self.evaluate(self.best_strategy, "dev").metric,
            "strategy": self.best_strategy,
            "strategy_id": strategy_key(self.best_strategy),
            "adapter": self.backend.active_adapter,
            "distill": distill,
        }
        if self.splits["test"]:
            record["test"] = self.test_report(self.best_strategy)
        self.logger.log("cycle", **record)
        return record

    def close(self) -> None:
        self.backend.close()
