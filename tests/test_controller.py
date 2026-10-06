import json

import pytest

from macd.backends.mock import MockBackend
from macd.core.controller import MetaController
from macd.core.evolution import strategy_key

from .conftest import MOCK_MODEL, qa_task


def test_scores_are_valid_and_reproducible(tmp_path):
    def run():
        ctrl = MetaController(task_cfg=qa_task(), model_cfg=MOCK_MODEL, seed=11)
        history = [ctrl.run_cycle() for _ in range(2)]
        return json.dumps(history, sort_keys=True, default=str)

    first = run()
    assert first == run()
    for record in json.loads(first):
        assert 0.0 <= record["val"] <= 1.0
        assert 0.0 <= record["test"]["metric"] <= 1.0


def test_evolution_never_loses_its_best_and_uses_offspring(controller):
    baseline = controller.evaluate_once()
    first = controller.evolve(population=6, top_k=2, generations=1)
    bred = {strategy_key(s) for s in first["population"]}
    second = controller.evolve(population=6, top_k=2, generations=1)
    assert first["best_metric"] >= baseline
    assert second["best_metric"] >= first["best_metric"]  # elitism
    assert len(first["population"]) == 6
    # offspring bred in the first call are what the second call scored
    scored_second = {key for key, _, split in controller._cache if split == "dev"}
    assert bred <= scored_second


def test_evaluation_is_cached(controller):
    controller.evaluate_once()
    calls = controller.backend.calls
    controller.evaluate_once()
    assert controller.backend.calls == calls


def test_test_split_is_never_used_for_selection(controller):
    controller.evolve(generations=2)
    controller.distill_if_improved(best_validation=controller.evaluate_once())
    assert not any(split == "test" for _, _, split in controller._cache)


def test_distillation_only_uses_train_data_and_respects_threshold(controller):
    evo = controller.evolve(generations=2)
    out = controller.distill_if_improved(best_validation=evo["best_metric"])
    train_prompts = {ex.prompt for ex in controller.splits["train"]}
    assert len(controller.store) > 0
    for row in controller.store.curated:
        assert row["prompt"] in train_prompts
        assert row["reward"] >= controller.task_cfg.training.distill_threshold
    assert out["curated"] == len(controller.store)
    assert out["dev_after"] >= out["dev_before"] or out["distilled"] is False


def test_no_second_distillation_without_improvement(controller):
    evo = controller.evolve(generations=1)
    controller.distill_if_improved(best_validation=evo["best_metric"])
    again = controller.distill_if_improved(best_validation=controller.best_validation)
    assert again == {"distilled": False, "reason": "no dev improvement"}


class HarmfulBackend(MockBackend):
    """Trains an adapter that makes every answer wrong."""

    def train(self, examples, hyper, out_dir, name, seed=0):
        self.memories[name] = {}
        return {}

    def _one(self, req):
        return "wrong" if self.active_adapter else super()._one(req)


def test_harmful_adapter_is_rolled_back():
    ctrl = MetaController(task_cfg=qa_task(), model_cfg=MOCK_MODEL, seed=7, backend=HarmfulBackend())
    evo = ctrl.evolve(generations=2)
    assert evo["best_metric"] > 0
    out = ctrl.distill_if_improved(best_validation=evo["best_metric"])
    assert out["distilled"] is False and "rolled back" in out["reason"]
    assert ctrl.backend.active_adapter is None
    assert ctrl.evaluate_once(evo["best_strategy"]) == evo["best_metric"]


def test_leakage_between_splits_is_rejected(tmp_path):
    dup = tmp_path / "dev.jsonl"
    dup.write_text('{"prompt": "Is water dry?", "answer": "no"}\n', encoding="utf-8")
    task = qa_task()
    task["task"]["data"]["dev_path"] = str(dup)
    with pytest.raises(ValueError, match="leakage"):
        MetaController(task_cfg=task, model_cfg=MOCK_MODEL)


def test_run_dir_gets_config_and_events(controller, tmp_path):
    controller.run_cycle()
    run = tmp_path / "run"
    assert json.loads((run / "config.json").read_text())["seed"] == 7
    events = [json.loads(line)["event"] for line in (run / "events.jsonl").read_text().splitlines()]
    assert "generation" in events and "cycle" in events
