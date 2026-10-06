"""Model backends. Pick one in the model YAML with ``model.backend``."""

from __future__ import annotations

from typing import Any

from .base import Backend, GenerationRequest, TrainingNotSupported


def build_backend(backend: str, params: dict[str, Any] | None = None) -> Backend:
    """Create a backend by name. Heavy backends are imported only when chosen."""
    if backend == "mock":
        from .mock import MockBackend

        return MockBackend(params)
    if backend == "openai_compat":
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(params)
    if backend == "hf":
        from .hf import HFBackend

        return HFBackend(params)
    raise ValueError(f"Unknown backend: {backend}")


__all__ = ["Backend", "GenerationRequest", "TrainingNotSupported", "build_backend"]
