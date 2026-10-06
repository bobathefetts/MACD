"""Validated configuration. YAML is parsed into these models, so a typo in a
config file fails loudly at startup instead of being silently ignored."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Strict):
    train_path: Path | None = None
    dev_path: Path
    test_path: Path | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_eval_path(cls, values: Any) -> Any:
        # v3.0 configs used a single `eval_path`
        if isinstance(values, dict) and "eval_path" in values:
            values = dict(values)
            values.setdefault("dev_path", values.pop("eval_path"))
        return values


class EvaluatorConfig(_Strict):
    metric: str = "default"


class TaskSection(_Strict):
    name: str = "task"
    type: Literal["qa", "summarization"]
    data: DataConfig
    evaluator: EvaluatorConfig = Field(default_factory=EvaluatorConfig)


class SearchConfig(_Strict):
    population: int = Field(8, ge=2)
    top_k: int = Field(3, ge=1)
    generations: int = Field(2, ge=1)
    mutation_strength: float = Field(0.15, gt=0, le=1)
    crossover_rate: float = Field(0.5, ge=0, le=1)


class TrainingConfig(_Strict):
    distill_threshold: float = Field(0.6, ge=0, le=1)
    min_improvement: float = Field(1e-3, ge=0)
    min_examples: int = Field(4, ge=1)
    regression_tolerance: float = Field(0.0, ge=0)


class TaskConfig(_Strict):
    task: TaskSection
    search: SearchConfig = Field(default_factory=SearchConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


class ModelSection(_Strict):
    name: str = "mock"
    backend: Literal["mock", "openai_compat", "hf"] = "mock"
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(_Strict):
    model: ModelSection = Field(default_factory=ModelSection)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_task_config(path: str | Path) -> TaskConfig:
    return TaskConfig.model_validate(_read_yaml(path))


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.model_validate(_read_yaml(path))


def coerce_task_config(cfg: Any) -> TaskConfig:
    """Accept a TaskConfig, a plain dict, or an OmegaConf object."""
    return cfg if isinstance(cfg, TaskConfig) else TaskConfig.model_validate(_to_dict(cfg))


def coerce_model_config(cfg: Any) -> ModelConfig:
    return cfg if isinstance(cfg, ModelConfig) else ModelConfig.model_validate(_to_dict(cfg))


def _to_dict(cfg: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg
    try:  # OmegaConf is optional; v3.0 callers passed DictConfig objects
        from omegaconf import OmegaConf

        return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        raise TypeError(f"Unsupported config type: {type(cfg)!r}") from None
