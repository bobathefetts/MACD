"""Command line entry point: ``macd`` or ``python -m macd.main``."""

from __future__ import annotations

import importlib
import json
import platform
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table
from typer import Option, Typer

from . import __version__
from .config import load_model_config, load_task_config
from .core.controller import MetaController
from .core.utils import seed_everything

app = Typer(add_completion=False, help="MACD v3 - Meta-Adaptive Context Distillation")
console = Console()

TASK_OPT = Option(..., help="Path to task YAML.")
MODEL_OPT = Option(..., help="Path to model YAML.")
SEED_OPT = Option(42, help="RNG seed.")
RUN_DIR_OPT = Option(None, help="Where to write logs and adapters. Default: outputs/run-<timestamp>.")


def _controller(task_config: str, model_config: str, seed: int, run_dir: str | None, log: bool) -> MetaController:
    seed_everything(seed)
    if log and run_dir is None:
        run_dir = str(Path("outputs") / time.strftime("run-%Y%m%d-%H%M%S"))
    return MetaController(
        task_cfg=load_task_config(task_config),
        model_cfg=load_model_config(model_config),
        seed=seed,
        run_dir=run_dir,
    )


@app.command()
def evaluate(
    task_config: str = TASK_OPT,
    model_config: str = MODEL_OPT,
    split: str = Option("dev", help="Which split to score: train, dev or test."),
    seed: int = SEED_OPT,
) -> None:
    """Score the default strategy once."""
    ctrl = _controller(task_config, model_config, seed, None, log=False)
    res = ctrl.evaluate(split=split)
    console.print({"split": split, "score": round(res.metric, 4), **res.details})
    ctrl.close()


@app.command()
def evolve(
    task_config: str = TASK_OPT,
    model_config: str = MODEL_OPT,
    population: int = Option(None, help="Population size (default: from task config)."),
    top_k: int = Option(None, help="Number of elites kept (default: from task config)."),
    generations: int = Option(None, help="Generations to run (default: from task config)."),
    seed: int = SEED_OPT,
    run_dir: str = RUN_DIR_OPT,
) -> None:
    """Run the strategy search only, without distillation."""
    ctrl = _controller(task_config, model_config, seed, run_dir, log=True)
    result = ctrl.evolve(population=population, top_k=top_k, generations=generations)
    console.print({"best_metric": round(result["best_metric"], 4), "best_strategy": result["best_strategy"]})
    ctrl.close()


@app.command()
def cycle(
    task_config: str = TASK_OPT,
    model_config: str = MODEL_OPT,
    cycles: int = Option(3, help="Number of MACD cycles."),
    population: int = Option(None, help="Population size (default: from task config)."),
    top_k: int = Option(None, help="Elites kept each generation (default: from task config)."),
    generations: int = Option(None, help="Generations per cycle (default: from task config)."),
    seed: int = SEED_OPT,
    run_dir: str = RUN_DIR_OPT,
) -> None:
    """Run full cycles: evolve, distill, report."""
    ctrl = _controller(task_config, model_config, seed, run_dir, log=True)
    baseline = ctrl.evaluate(split="dev").metric
    table = Table(title=f"MACD run ({ctrl.model_cfg.model.backend} backend)")
    for col in ("cycle", "dev", "test (95% CI)", "adapter", "distillation", "style", "shots"):
        table.add_column(col)
    table.add_row("baseline", f"{baseline:.3f}", "-", "-", "-", ctrl.best_strategy["prompt_style"], "0")

    history = []
    for c in range(1, cycles + 1):
        console.print(f"[bold cyan]=== CYCLE {c}/{cycles} ===[/]")
        record = ctrl.run_cycle(population=population, top_k=top_k, generations=generations)
        history.append({"cycle": c, **record})
        test = record.get("test")
        test_text = f"{test['metric']:.3f} ({test['ci95'][0]:.2f}-{test['ci95'][1]:.2f})" if test else "-"
        table.add_row(
            str(c),
            f"{record['val']:.3f}",
            test_text,
            record["adapter"] or "base",
            record["distill"]["reason"],
            record["strategy"]["prompt_style"],
            str(record["strategy"]["retrieval"]["k"]),
        )
    console.print(table)
    if ctrl.logger.run_dir:
        out = ctrl.logger.run_dir / "history.json"
        out.write_text(json.dumps({"baseline_dev": baseline, "history": history}, indent=2), encoding="utf-8")
        console.print(f"Run saved to [bold]{ctrl.logger.run_dir}[/]")
    ctrl.close()


@app.command()
def doctor() -> None:
    """Check the environment: Python, packages, GPU."""
    table = Table(title=f"MACD {__version__} environment")
    table.add_column("item")
    table.add_column("status")
    table.add_row("python", platform.python_version())
    for name, needed_for in [
        ("typer", "core"),
        ("pydantic", "core"),
        ("yaml", "core"),
        ("httpx", "openai_compat backend"),
        ("torch", "hf backend"),
        ("transformers", "hf backend"),
        ("peft", "hf backend (LoRA)"),
        ("bitsandbytes", "hf backend (4-bit, optional)"),
    ]:
        try:
            module = importlib.import_module(name)
            table.add_row(name, f"[green]{getattr(module, '__version__', 'installed')}[/]")
        except ImportError:
            table.add_row(name, f"[yellow]missing[/] (needed for {needed_for})")
    try:
        import torch

        if torch.cuda.is_available():
            gpus = ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
            table.add_row("gpu", f"[green]{gpus}[/]")
        else:
            table.add_row("gpu", "[yellow]no CUDA device; hf backend will run on CPU[/]")
    except ImportError:
        table.add_row("gpu", "unknown (torch not installed)")
    console.print(table)


if __name__ == "__main__":
    app()
