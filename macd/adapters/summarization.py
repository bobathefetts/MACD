"""One-sentence summarization."""

from __future__ import annotations

from ..core.evaluator import SummarizationEvaluator
from .base import TaskAdapter


class SummarizationAdapter(TaskAdapter):
    task_type = "summarization"
    task_instruction = "Summarize the text in one short sentence."

    def __init__(self) -> None:
        self.evaluator = SummarizationEvaluator()
