"""Minimal run logging: one JSON line per event, plus a config snapshot."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    """Writes ``events.jsonl`` and ``config.json`` under ``run_dir``.

    With ``run_dir=None`` it records events in memory only, which is what the
    tests use.
    """

    def __init__(self, run_dir: str | Path | None = None) -> None:
        self.run_dir = Path(run_dir) if run_dir else None
        self.events: list[dict[str, Any]] = []
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: dict[str, Any]) -> None:
        if self.run_dir:
            (self.run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    def log(self, event: str, **fields: Any) -> None:
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **fields}
        self.events.append(record)
        if self.run_dir:
            with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
