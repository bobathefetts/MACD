from pathlib import Path

import pytest
from typer.testing import CliRunner

from macd.main import app

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


@pytest.fixture(autouse=True)
def _in_repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)


@pytest.mark.parametrize("task", ["qa", "summarization"])
def test_cycle_command(task, tmp_path):
    result = runner.invoke(
        app,
        [
            "cycle",
            "--task-config",
            f"configs/tasks/{task}.yaml",
            "--model-config",
            "configs/models/local.yaml",
            "--cycles",
            "2",
            "--run-dir",
            str(tmp_path / "run"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "run" / "history.json").exists()


def test_evaluate_and_doctor_commands():
    args = ["--task-config", "configs/tasks/qa.yaml", "--model-config", "configs/models/local.yaml"]
    assert runner.invoke(app, ["evaluate", *args, "--split", "test"]).exit_code == 0
    assert runner.invoke(app, ["doctor"]).exit_code == 0
