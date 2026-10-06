"""Shared adapter machinery.

An adapter turns a dataset row plus a strategy into chat messages, and turns
the raw model output back into a bare answer that the evaluator can score.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.evaluator import Evaluator

Message = dict[str, str]

STYLE_INSTRUCTIONS = {
    "concise": "Reply with only the answer and nothing else.",
    "cot": "Think step by step, then end with a line of the form 'Final answer: <answer>'.",
    "bullet": "Reply with a single bullet point that starts with '- ' and contains only the answer.",
    "json": 'Reply with a single JSON object of the form {"answer": "<answer>"} and nothing else.',
}

_FINAL_RE = re.compile(r"final answer\s*[:\-]\s*(.*)", re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str


def format_answer(answer: str, style: str) -> str:
    """Render a gold answer the way the model is asked to answer in ``style``."""
    if style == "cot":
        return f"Final answer: {answer}"
    if style == "bullet":
        return f"- {answer}"
    if style == "json":
        return json.dumps({"answer": answer})
    return answer


def extract_answer(raw: str, style: str) -> str:
    """Pull the bare answer out of a styled model response.

    Returns an empty string when the required format is missing, so a strategy
    whose output cannot be parsed scores zero rather than getting lucky.
    """
    text = raw.strip()
    if style == "json":
        match = _JSON_RE.search(text)
        if not match:
            return ""
        try:
            value = json.loads(match.group(0)).get("answer", "")
        except (json.JSONDecodeError, AttributeError):
            return ""
        return str(value).strip()
    if style == "cot":
        matches = _FINAL_RE.findall(text)
        return matches[-1].strip() if matches else ""
    if style == "bullet":
        for line in text.splitlines():
            if line.strip().startswith(("-", "*", "•")):
                return line.strip().lstrip("-*• ").strip()
        return ""
    return text


class TaskAdapter:
    """Base class. Subclasses set ``task_instruction`` and ``evaluator``."""

    task_type: str = ""
    task_instruction: str = ""
    evaluator: Evaluator

    def load(self, path: str | Path) -> list[Example]:
        items: list[Example] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj: dict[str, Any] = json.loads(line)
                if "prompt" not in obj or "answer" not in obj:
                    raise ValueError(f"{path}:{line_no} needs 'prompt' and 'answer' fields")
                items.append(Example(prompt=str(obj["prompt"]), answer=str(obj["answer"])))
        return items

    def build_messages(self, example: Example, style: str, exemplars: list[Example] | None = None) -> list[Message]:
        system = f"{self.task_instruction} {STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS['concise'])}"
        messages: list[Message] = [{"role": "system", "content": system}]
        for shot in exemplars or []:
            messages.append({"role": "user", "content": shot.prompt})
            messages.append({"role": "assistant", "content": format_answer(shot.answer, style)})
        messages.append({"role": "user", "content": example.prompt})
        return messages

    def extract(self, raw: str, style: str) -> str:
        return extract_answer(raw, style)
