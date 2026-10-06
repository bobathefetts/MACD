from pathlib import Path

import pytest

from macd.core.controller import MetaController

ROOT = Path(__file__).resolve().parents[1]


def qa_task(**overrides):
    cfg = {
        "task": {
            "name": "toy_qa",
            "type": "qa",
            "data": {
                "train_path": str(ROOT / "data/qa/train.jsonl"),
                "dev_path": str(ROOT / "data/qa/dev.jsonl"),
                "test_path": str(ROOT / "data/qa/test.jsonl"),
            },
        },
    }
    cfg.update(overrides)
    return cfg


MOCK_MODEL = {"model": {"name": "mock", "backend": "mock"}}


@pytest.fixture
def controller(tmp_path):
    return MetaController(task_cfg=qa_task(), model_cfg=MOCK_MODEL, seed=7, run_dir=tmp_path / "run")
