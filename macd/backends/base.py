"""Backend interface. The controller only ever talks to this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Message = dict[str, str]


class TrainingNotSupported(RuntimeError):
    """Raised when distillation is requested from a backend that cannot train."""


@dataclass
class GenerationRequest:
    messages: list[Message]
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 64
    seed: int = 0
    # Free-form hints (prompt style, task type). Real backends ignore them;
    # the mock backend uses them to format its output.
    hints: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    """Generates text and, optionally, trains and swaps LoRA adapters."""

    supports_training: bool = False

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})
        self.active_adapter: str | None = None

    @abstractmethod
    def generate(self, requests: list[GenerationRequest]) -> list[str]:
        """Return one completion per request, in order."""

    def train(
        self, examples: list[dict[str, Any]], hyper: dict[str, Any], out_dir: Path, name: str, seed: int = 0
    ) -> dict[str, Any]:
        """Train adapter ``name`` on ``examples`` and save it under ``out_dir``.

        Each example has ``messages`` (chat prompt), ``prompt`` (raw input) and
        ``target`` (the text the model should produce). Must not change which
        adapter is active.
        """
        raise TrainingNotSupported(f"{type(self).__name__} cannot train adapters")

    def activate(self, adapter: str | None) -> None:
        """Switch generation to ``adapter``; ``None`` means the base model."""
        self.active_adapter = adapter

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""
