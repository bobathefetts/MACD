"""Task adapters: data loading, prompt construction and answer extraction."""

from __future__ import annotations

from .base import Example, TaskAdapter, extract_answer, format_answer
from .qa import QAAdapter
from .summarization import SummarizationAdapter

ADAPTERS: dict[str, type[TaskAdapter]] = {
    "qa": QAAdapter,
    "summarization": SummarizationAdapter,
}


def get_adapter(task_type: str) -> TaskAdapter:
    try:
        return ADAPTERS[task_type]()
    except KeyError:
        raise ValueError(f"Unsupported task type: {task_type}") from None


__all__ = ["ADAPTERS", "Example", "TaskAdapter", "extract_answer", "format_answer", "get_adapter"]
