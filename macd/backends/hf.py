"""In-process Hugging Face backend: batched generation plus real LoRA training.

Heavy imports (torch, transformers, peft) happen lazily, so the rest of the
package works without them. Install with ``pip install -e ".[hf]"``.

Model params (all optional except ``model_id``):

    model_id: meta-llama/Llama-3.1-8B-Instruct
    device_map: auto            # or "cpu"
    dtype: auto                 # auto | float16 | bfloat16 | float32
    load_in_4bit: true          # QLoRA-style 4-bit loading; needs CUDA + bitsandbytes
    batch_size: 8               # generation batch size
    train_batch_size: 4
    max_seq_len: 1024
    lora_alpha_ratio: 2.0       # alpha = ratio * rank
    lora_dropout: 0.05
    target_modules: all-linear  # or a list such as [q_proj, v_proj]
    local_files_only: false     # set true for air-gapped machines
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .base import Backend, GenerationRequest, Message


class HFBackend(Backend):
    supports_training = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        if "model_id" not in self.params:
            raise ValueError("HF backend needs `params.model_id`")
        self.model: Any = None
        self.tokenizer: Any = None
        self._has_peft = False

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError('The HF backend needs extra packages: pip install -e ".[hf]"') from exc

        p = self.params
        local = bool(p.get("local_files_only", False))
        self.tokenizer = AutoTokenizer.from_pretrained(p["model_id"], local_files_only=local)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # required for batched generation

        kwargs: dict[str, Any] = {"local_files_only": local}
        dtype = p.get("dtype", "auto")
        kwargs["dtype"] = dtype if dtype == "auto" else getattr(torch, dtype)
        if p.get("device_map", "auto") != "cpu":
            kwargs["device_map"] = p.get("device_map", "auto")
        self._quantized = bool(p.get("load_in_4bit", False)) and torch.cuda.is_available()
        if self._quantized:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        self.model = AutoModelForCausalLM.from_pretrained(p["model_id"], **kwargs)
        self.model.eval()

    def _render(self, messages: list[Message]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        lines = [f"{m['role']}: {m['content']}" for m in messages]
        return "\n".join(lines) + "\nassistant:"

    def _encode(self, text: str) -> list[int]:
        # chat templates already contain the special tokens
        add_special = not getattr(self.tokenizer, "chat_template", None)
        return self.tokenizer(text, add_special_tokens=add_special)["input_ids"]

    # -------------------------------------------------------------- generate
    def generate(self, requests: list[GenerationRequest]) -> list[str]:
        import torch

        self._load()
        outputs: list[str] = [""] * len(requests)
        # group requests that share sampling settings so they can be batched
        groups: dict[tuple[float, float, int], list[int]] = {}
        for i, r in enumerate(requests):
            groups.setdefault((r.temperature, r.top_p, r.max_new_tokens), []).append(i)
        batch_size = int(self.params.get("batch_size", 8))
        for (temperature, top_p, max_new_tokens), indices in groups.items():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start : start + batch_size]
                ids = [self._encode(self._render(requests[i].messages)) for i in chunk]
                width = max(len(x) for x in ids)
                pad = self.tokenizer.pad_token_id
                input_ids = torch.tensor([[pad] * (width - len(x)) + x for x in ids])
                mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in ids])
                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": pad,
                    "do_sample": temperature > 0,
                }
                if temperature > 0:
                    gen_kwargs.update(temperature=temperature, top_p=top_p)
                torch.manual_seed(requests[chunk[0]].seed % (2**31))
                with torch.inference_mode():
                    out = self.model.generate(
                        input_ids=input_ids.to(self.model.device),
                        attention_mask=mask.to(self.model.device),
                        **gen_kwargs,
                    )
                for row, i in enumerate(chunk):
                    outputs[i] = self.tokenizer.decode(out[row][width:], skip_special_tokens=True).strip()
        return outputs

    # ----------------------------------------------------------------- train
    def train(
        self, examples: list[dict[str, Any]], hyper: dict[str, Any], out_dir: Path, name: str, seed: int = 0
    ) -> dict[str, Any]:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        self._load()
        p = self.params
        rank = int(hyper.get("rank", 8))
        steps = int(hyper.get("steps", 100))
        lr = float(hyper.get("lr", 1e-4))
        config = LoraConfig(
            r=rank,
            lora_alpha=int(rank * float(p.get("lora_alpha_ratio", 2.0))),
            lora_dropout=float(p.get("lora_dropout", 0.05)),
            target_modules=p.get("target_modules", "all-linear"),
            task_type="CAUSAL_LM",
        )
        previous = self.active_adapter
        if not self._has_peft:
            if self._quantized:
                self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, config, adapter_name=name)
            self._has_peft = True
        else:
            self.model.enable_adapter_layers()
            self.model.add_adapter(name, config)
        self.model.set_adapter(name)

        # tokenise once; loss is computed on the target tokens only
        max_len = int(p.get("max_seq_len", 1024))
        eos = self.tokenizer.eos_token or ""
        rows: list[tuple[list[int], list[int]]] = []
        for ex in examples:
            prompt_ids = self._encode(self._render(ex["messages"]))
            target_ids = self.tokenizer(" " + ex["target"] + eos, add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + target_ids)[-max_len:]
            n_prompt = len(ids) - len(target_ids)
            rows.append((ids, [-100] * max(n_prompt, 0) + ids[max(n_prompt, 0) :]))

        params = [w for w in self.model.parameters() if w.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr)
        rng = random.Random(seed)
        batch_size = min(int(p.get("train_batch_size", 4)), len(rows))
        pad = self.tokenizer.pad_token_id
        order: list[int] = []
        losses: list[float] = []
        self.model.train()
        try:
            for _ in range(steps):
                if len(order) < batch_size:
                    fresh = list(range(len(rows)))
                    rng.shuffle(fresh)
                    order.extend(fresh)
                batch = [rows[order.pop()] for _ in range(batch_size)]
                width = max(len(ids) for ids, _ in batch)
                input_ids = torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in batch])
                labels = torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in batch])
                mask = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch])
                device = self.model.device
                loss = self.model(
                    input_ids=input_ids.to(device), attention_mask=mask.to(device), labels=labels.to(device)
                ).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
        finally:
            self.model.eval()
            self.model.save_pretrained(str(out_dir), selected_adapters=[name])
            # PEFT writes named adapters to <out_dir>/<name>/; flatten so out_dir
            # is directly loadable by PEFT, vLLM and friends
            nested = Path(out_dir) / name
            if nested.is_dir():
                for item in nested.iterdir():
                    item.replace(Path(out_dir) / item.name)
                nested.rmdir()
            self.activate(previous)  # training must not change the active adapter

        tail = losses[-max(1, len(losses) // 10) :]
        return {
            "first_loss": losses[0] if losses else None,
            "final_loss": sum(tail) / len(tail) if tail else None,
            "steps": steps,
            "rank": rank,
            "lr": lr,
            "trainable_params": sum(w.numel() for w in params),
        }

    def activate(self, adapter: str | None) -> None:
        if self._has_peft:
            if adapter is None:
                self.model.disable_adapter_layers()
            else:
                self.model.enable_adapter_layers()
                self.model.set_adapter(adapter)
        elif adapter is not None:
            raise ValueError(f"Unknown adapter: {adapter}")
        super().activate(adapter)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
