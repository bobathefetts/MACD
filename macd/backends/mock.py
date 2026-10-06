"""Mock backend for tests and CI. No GPU, no network, no downloads.

It is a tiny but honest learner: a nearest-neighbour memoriser. It answers by
copying the answer of the most similar example it can see, either among the
few-shot exemplars in the prompt or in the memory written by ``train``. With
nothing similar in view it guesses. That is enough for every gene in a
strategy to have a measurable effect, and for distillation to change behaviour
for a real reason.

Scores from this backend say nothing about language models. It exists to test
the plumbing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..adapters.base import extract_answer, format_answer
from ..core.retrieval import jaccard
from ..core.utils import stable_hash, stable_unit
from .base import Backend, GenerationRequest

_GUESSES = ("yes", "no", "maybe")
_REASONING = "Let me think about this carefully and work through it one step at a time before I answer."


class MockBackend(Backend):
    supports_training = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.match_threshold = float(self.params.get("match_threshold", 0.5))
        self.memories: dict[str, dict[str, str]] = {}
        self.calls = 0

    def generate(self, requests: list[GenerationRequest]) -> list[str]:
        self.calls += len(requests)
        return [self._one(r) for r in requests]

    def _one(self, req: GenerationRequest) -> str:
        style = req.hints.get("style", "concise")
        turns = [m for m in req.messages if m["role"] != "system"]
        query = turns[-1]["content"]
        known: list[tuple[str, str]] = []
        for user, assistant in zip(turns[:-1:2], turns[1:-1:2], strict=False):
            known.append((user["content"], extract_answer(assistant["content"], style)))
        if self.active_adapter:
            known.extend(self.memories.get(self.active_adapter, {}).items())

        best_sim, answer = 0.0, ""
        for prompt, ans in known:
            sim = jaccard(prompt, query)
            if sim > best_sim:
                best_sim, answer = sim, ans
        if best_sim < self.match_threshold:
            if req.hints.get("task") == "summarization":
                answer = " ".join(query.split()[:8])
            else:
                answer = _GUESSES[stable_hash(query) % len(_GUESSES)]
        # sampling noise: hotter and wider sampling derails more answers
        if stable_unit(query, req.seed, "noise") < 0.3 * req.temperature * req.top_p:
            answer = "unsure"

        text = format_answer(answer, style)
        if style == "cot":
            text = f"{_REASONING} {text}"
        # a token budget that is too small cuts the answer off
        return " ".join(text.split()[: req.max_new_tokens])

    def train(
        self, examples: list[dict[str, Any]], hyper: dict[str, Any], out_dir: Path, name: str, seed: int = 0
    ) -> dict[str, Any]:
        memory = {ex["prompt"]: ex["target"] for ex in examples}
        self.memories[name] = memory
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "mock_adapter.json").write_text(json.dumps(memory, indent=2), encoding="utf-8")
        return {"memorised": len(memory)}
