"""Short-answer question answering."""

from __future__ import annotations

from ..core.evaluator import QAEvaluator
from .base import TaskAdapter


class QAAdapter(TaskAdapter):
    task_type = "qa"
    task_instruction = "Answer the question as briefly as possible."

    def __init__(self) -> None:
        self.evaluator = QAEvaluator()
