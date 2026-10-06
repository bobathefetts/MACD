from pathlib import Path

import pytest
from pydantic import ValidationError

from macd.config import TaskConfig, load_model_config, load_task_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["qa.yaml", "summarization.yaml"])
def test_shipped_task_configs_load(name):
    cfg = load_task_config(ROOT / "configs/tasks" / name)
    assert cfg.task.data.train_path is not None


@pytest.mark.parametrize("path", sorted((ROOT / "configs/models").glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_model_configs_load(path):
    assert load_model_config(path).model.backend in {"mock", "openai_compat", "hf"}


def test_legacy_eval_path_is_accepted():
    cfg = TaskConfig.model_validate({"task": {"type": "qa", "data": {"eval_path": "data/qa/dev.jsonl"}}})
    assert cfg.task.data.dev_path == Path("data/qa/dev.jsonl")


def test_typos_fail_loudly():
    with pytest.raises(ValidationError):
        TaskConfig.model_validate({"task": {"type": "qa", "data": {"dev_path": "x"}}, "search": {"populaton": 8}})
    with pytest.raises(ValidationError):
        TaskConfig.model_validate({"task": {"type": "poetry", "data": {"dev_path": "x"}}})
