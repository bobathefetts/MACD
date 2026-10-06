"""Curated-sample store and the trainer that hands it to a backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..backends.base import Backend


@dataclass
class DistillationStore:
    """High-reward samples, de-duplicated by prompt (best reward wins)."""

    _by_prompt: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, prompt: str, prediction: str, reference: str, reward: float, threshold: float = 0.6) -> bool:
        if reward < threshold:
            return False
        current = self._by_prompt.get(prompt)
        if current is not None and current["reward"] >= reward:
            return False
        self._by_prompt[prompt] = {
            "prompt": prompt,
            "prediction": prediction,
            "reference": reference,
            "reward": reward,
        }
        return True

    @property
    def curated(self) -> list[dict[str, Any]]:
        return list(self._by_prompt.values())

    def __len__(self) -> int:
        return len(self._by_prompt)


class Trainer:
    """Distills curated samples into an adapter using the backend's trainer."""

    def __init__(self, backend: Backend, out_dir: str | Path = "outputs/adapters") -> None:
        self.backend = backend
        self.out_dir = Path(out_dir)
        self.rounds = 0

    def distill(self, examples: list[dict[str, Any]], hyper: dict[str, Any], seed: int = 0) -> dict[str, Any]:
        """Train a fresh adapter on ``examples``. Does not activate it."""
        self.rounds += 1
        name = f"macd-r{self.rounds}"
        out = self.out_dir / name
        out.mkdir(parents=True, exist_ok=True)
        info = self.backend.train(examples, hyper=hyper, out_dir=out, name=name, seed=seed)
        return {"adapter": name, "path": str(out), "examples_used": len(examples), **info}
