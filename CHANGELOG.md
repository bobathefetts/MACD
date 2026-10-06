# Changelog

## 3.1.0

A rewrite of the 3.0.0 code published here. The 3.0.0 files remain available in the git history.

### Fixed
- The package now imports and runs. 3.0.0 failed at import under its own pinned pydantic version.
- Evolution works. Offspring were bred and then discarded; the population now persists between generations and cycles, and `crossover` is used.
- Runs are reproducible. Python's per-process `hash()` and unseeded RNGs are replaced with stable hashing and one seeded RNG.
- Train/dev leakage is gone. The toy data files were identical and malformed; there are now disjoint train, dev and test splits, and the controller rejects overlapping splits.
- The simulated `memory_boost` improvement, which nudged the mock model toward "yes", is removed.
- Config values (`search`, `training.distill_threshold`) are read and validated instead of ignored.

### Added
- Switchable backends: `mock`, `openai_compat` (vLLM, Ollama, llama.cpp and other OpenAI-style servers), and `hf` (Transformers with PEFT LoRA training and optional 4-bit loading).
- Adapter gating: a distilled adapter is kept only if the dev score does not regress; otherwise it is rolled back.
- Every strategy gene has an effect: few-shot retrieval (`k`, `selector`), sampling settings, prompt style with strict answer extraction, and LoRA hyperparameters.
- Chat-template prompting, batched generation, and loss computed on answer tokens only in the Hugging Face backend.
- Held-out test reporting with 95% bootstrap confidence intervals.
- Evaluation cache, run logging (`config.json`, `events.jsonl`, `history.json`), and a `macd doctor` environment check.
- Summarization toy data.
- `pyproject.toml` with optional extras, a `macd` console command, ruff, GitHub Actions CI, contributing guide, MIT license file.

### Changed
- The project now lives at the repository root instead of a `macd_v3/` subfolder.
- The README documents only what the code does. Earlier text described planned features (entropy-based token pruning, latent-space alignment, multi-GPU orchestration, a context-reduction figure) that were never implemented or measured.
- Package layout is `macd/core`, `macd/backends`, `macd/adapters`. The `macd/models`, `macd/config/` and `macd/logging` packages are replaced.
- The `openai` backend is replaced by `openai_compat`, which talks to any OpenAI-style chat endpoint without the `openai` package.
- Dependencies are down to typer, pydantic, pyyaml, rich and httpx for the core. `omegaconf`, `loguru`, `scikit-learn`, `scipy`, `datasets`, `openai` and `wandb` are no longer required.
- Python 3.10 or newer is required.

### Removed
- Weights & Biases experiment tracker (runs are logged to local JSON files instead).
- Status documents from development (`CRITICAL_REASSESSMENT.md`, `FIXES_APPLIED.md`, `GPU_FIXES_SUMMARY.md`, `GPU_CONFIGURATION.md`, `TEST_RESULTS.md`, `MIGRATION.md`, `INSTALLATION_AND_TESTING.md`) and ad-hoc scripts (`test_fixes.py`, `validate_setup.py` and similar).
- Committed `__pycache__` files and local editor settings.
- The `retrieval.chunk_size` and `repetition_penalty` genes, which nothing consumed.
