"""Backend for any OpenAI-compatible chat endpoint (vLLM, Ollama, llama.cpp
server, LM Studio, or a hosted API).

Generation runs against the endpoint with a thread pool. The endpoint itself
cannot train, so distillation is optional: set ``params.trainer`` to an HF
backend config and adapters are trained locally with PEFT, then registered
with the server through vLLM's runtime LoRA API if ``params.lora_load`` is on.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from .base import Backend, GenerationRequest, TrainingNotSupported

_RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class OpenAICompatBackend(Backend):
    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        p = self.params
        self.model = p.get("model", "default")
        self.concurrency = int(p.get("concurrency", 8))
        self.max_retries = int(p.get("max_retries", 3))
        self.send_seed = bool(p.get("send_seed", True))
        self.extra_body: dict[str, Any] = dict(p.get("extra_body") or {})
        self.trainer_params: dict[str, Any] | None = p.get("trainer")
        self.lora_load = bool(p.get("lora_load", False))
        self.supports_training = self.trainer_params is not None

        headers = {}
        api_key = os.environ.get(p.get("api_key_env", "MACD_API_KEY"), "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=str(p.get("base_url", "http://localhost:8000/v1")).rstrip("/") + "/",
            headers=headers,
            timeout=float(p.get("timeout", 120)),
            transport=p.get("transport"),  # injectable for tests
        )

    def generate(self, requests: list[GenerationRequest]) -> list[str]:
        if len(requests) <= 1 or self.concurrency <= 1:
            return [self._one(r) for r in requests]
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(self._one, requests))

    def _one(self, req: GenerationRequest) -> str:
        body: dict[str, Any] = {
            "model": self.active_adapter or self.model,
            "messages": req.messages,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_new_tokens,
            **self.extra_body,
        }
        if self.send_seed:
            body["seed"] = req.seed % (2**31)
        data = self._post("chat/completions", body)
        return data["choices"][0]["message"].get("content") or ""

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(min(2 ** (attempt - 1), 8) * float(self.params.get("retry_backoff", 1.0)))
            try:
                resp = self._client.post(path, json=body)
            except httpx.TransportError as exc:
                last_error = exc
                continue
            if resp.status_code in _RETRY_STATUS:
                last_error = RuntimeError(f"HTTP {resp.status_code}")
                continue
            if resp.is_error:  # 4xx: retrying will not help
                raise RuntimeError(f"Request to {path} failed: HTTP {resp.status_code} {resp.text[:300]}")
            return resp.json() if resp.content else {}
        raise RuntimeError(f"Request to {path} failed after {self.max_retries + 1} attempts: {last_error}")

    def train(
        self, examples: list[dict[str, Any]], hyper: dict[str, Any], out_dir: Path, name: str, seed: int = 0
    ) -> dict[str, Any]:
        if self.trainer_params is None:
            raise TrainingNotSupported(
                "This endpoint backend has no trainer. Add `params.trainer` (an HF config) to enable distillation."
            )
        from .hf import HFBackend

        trainer = HFBackend(self.trainer_params)
        try:
            info = trainer.train(examples, hyper=hyper, out_dir=out_dir, name=name, seed=seed)
        finally:
            trainer.close()
        if self.lora_load:
            # vLLM: start the server with --enable-lora and
            # VLLM_ALLOW_RUNTIME_LORA_UPDATING=True for this call to work.
            self._post("load_lora_adapter", {"lora_name": name, "lora_path": str(Path(out_dir).resolve())})
            info["registered_with_server"] = True
        return info

    def activate(self, adapter: str | None) -> None:
        if adapter is not None and not self.lora_load:
            raise TrainingNotSupported(
                "Adapter was trained but the endpoint cannot load it. Set `params.lora_load: true` "
                "(vLLM runtime LoRA) or serve the adapter yourself."
            )
        super().activate(adapter)

    def close(self) -> None:
        self._client.close()
