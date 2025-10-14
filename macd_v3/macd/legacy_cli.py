"""
Legacy CLI commands for backward compatibility.
These commands support the old separate task and model config format.
"""

import warnings
from typer import Typer, Option, Exit
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional

from .core.controller import MetaController
from .core.utils import seed_everything
from .core.exceptions import MACDError, ConfigurationError, ValidationError
from .config.legacy import LegacyConfigAdapter
from .logging import setup_logger

# Initialize console and app
console = Console()
app = Typer(
    name="macd-legacy",
    help="MACD v3 Legacy CLI - Backward compatibility for old config format",
    add_completion=False,
    rich_markup_mode="rich"
)

@app.command()
def evaluate(
    task_config: str = Option(..., help="Path to task YAML configuration file."),
    model_config: str = Option(..., help="Path to model YAML configuration file."),
    seed: int = Option(42, help="Random number generator seed for reproducibility."),
    output_format: str = Option("json", help="Output format: 'json' or 'table'."),
    verbose: bool = Option(False, help="Enable verbose logging."),
):
    """
    [DEPRECATED] Evaluate a model on a task using the current best strategy.
    
    This command is deprecated. Please use the new unified config format:
    python -m macd.main evaluate --config config.yaml
    
    This command loads the specified task and model configurations,
    initializes a MetaController, and evaluates the model's performance
    on the task using the current best strategy.
    """
    try:
        # Show deprecation warning
        console.print(Panel(
            "[yellow]⚠️  DEPRECATION WARNING[/yellow]\n\n"
            "This command uses the legacy configuration format.\n"
            "Please update to the new unified format:\n\n"
            "python -m macd.main evaluate --config config.yaml\n\n"
            "See the migration guide for details.",
            title="Legacy Command",
            border_style="yellow"
        ))
        
        # Setup logging
        if verbose:
            setup_logger(log_level="DEBUG")
        else:
            setup_logger(log_level="INFO")
        
        # Set random seed
        seed_everything(seed)
        
        # Load and convert legacy configurations
        console.print(f"[blue]Loading legacy configurations...[/blue]")
        unified_config = LegacyConfigAdapter.load_legacy_configs(task_config, model_config)
        
        # Create controller
        console.print(f"[blue]Initializing MetaController...[/blue]")
        ctrl = MetaController(task_cfg=unified_config, model_cfg=unified_config)
        
        # Run evaluation
        console.print(f"[blue]Running evaluation...[/blue]")
        score = ctrl.evaluate_once()
        
        # Display results
        if output_format == "table":
            table = Table(title="Evaluation Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Score", f"{score:.4f}")
            console.print(table)
        else:
            console.print(f"{{'score': {score:.4f}}}")
        
        console.print(f"[green]✅ Evaluation completed successfully![/green]")
        
    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found: {e}[/red]")
        raise Exit(1)
    except ConfigurationError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise Exit(1)
    except ValidationError as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise Exit(1)
    except MACDError as e:
        console.print(f"[red]MACD error: {e}[/red]")
        raise Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise Exit(1)

@app.command()
def evolve(
    task_config: str = Option(..., help="Path to task YAML configuration file."),
    model_config: str = Option(..., help="Path to model YAML configuration file."),
    population: int = Option(8, help="Population size for evolution."),
    top_k: int = Option(3, help="Number of elite strategies to keep."),
    seed: int = Option(42, help="Random number generator seed for reproducibility."),
    output_format: str = Option("json", help="Output format: 'json' or 'table'."),
    verbose: bool = Option(False, help="Enable verbose logging."),
):
    """
    [DEPRECATED] Run evolution to find better strategies.
    
    This command is deprecated. Please use the new unified config format:
    python -m macd.main evolve --config config.yaml
    """
    try:
        # Show deprecation warning
        console.print(Panel(
            "[yellow]⚠️  DEPRECATION WARNING[/yellow]\n\n"
            "This command uses the legacy configuration format.\n"
            "Please update to the new unified format:\n\n"
            "python -m macd.main evolve --config config.yaml\n\n"
            "See the migration guide for details.",
            title="Legacy Command",
            border_style="yellow"
        ))
        
        # Setup logging
        if verbose:
            setup_logger(log_level="DEBUG")
        else:
            setup_logger(log_level="INFO")
        
        # Set random seed
        seed_everything(seed)
        
        # Load and convert legacy configurations
        console.print(f"[blue]Loading legacy configurations...[/blue]")
        unified_config = LegacyConfigAdapter.load_legacy_configs(task_config, model_config)
        
        # Create controller
        console.print(f"[blue]Initializing MetaController...[/blue]")
        ctrl = MetaController(task_cfg=unified_config, model_cfg=unified_config)
        
        # Run evolution
        console.print(f"[blue]Running evolution (population={population}, top_k={top_k})...[/blue]")
        result = ctrl.evolve(population=population, top_k=top_k)
        
        # Display results
        if output_format == "table":
            table = Table(title="Evolution Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Best Score", f"{result['best_score']:.4f}")
            table.add_row("Population Size", str(len(result['population'])))
            table.add_row("Elite Size", str(len(result['elites'])))
            console.print(table)
        else:
            console.print(f"{{'best_score': {result['best_score']:.4f}, 'population_size': {len(result['population'])}}}")
        
        console.print(f"[green]✅ Evolution completed successfully![/green]")
        
    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found: {e}[/red]")
        raise Exit(1)
    except ConfigurationError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise Exit(1)
    except ValidationError as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise Exit(1)
    except MACDError as e:
        console.print(f"[red]MACD error: {e}[/red]")
        raise Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise Exit(1)

@app.command()
def cycle(
    task_config: str = Option(..., help="Path to task YAML configuration file."),
    model_config: str = Option(..., help="Path to model YAML configuration file."),
    cycles: int = Option(3, help="Number of MACD cycles to run."),
    population: int = Option(8, help="Population size for evolution."),
    top_k: int = Option(3, help="Number of elite strategies to keep."),
    seed: int = Option(42, help="Random number generator seed for reproducibility."),
    output_format: str = Option("json", help="Output format: 'json' or 'table'."),
    verbose: bool = Option(False, help="Enable verbose logging."),
    save_results: bool = Option(False, help="Save results to file."),
    results_file: str = Option("results.json", help="File to save results to."),
):
    """
    [DEPRECATED] Run complete MACD cycles with evolution, evaluation, and distillation.
    
    This command is deprecated. Please use the new unified config format:
    python -m macd.main cycle --config config.yaml
    """
    try:
        # Show deprecation warning
        console.print(Panel(
            "[yellow]⚠️  DEPRECATION WARNING[/yellow]\n\n"
            "This command uses the legacy configuration format.\n"
            "Please update to the new unified format:\n\n"
            "python -m macd.main cycle --config config.yaml\n\n"
            "See the migration guide for details.",
            title="Legacy Command",
            border_style="yellow"
        ))
        
        # Setup logging
        if verbose:
            setup_logger(log_level="DEBUG")
        else:
            setup_logger(log_level="INFO")
        
        # Set random seed
        seed_everything(seed)
        
        # Load and convert legacy configurations
        console.print(f"[blue]Loading legacy configurations...[/blue]")
        unified_config = LegacyConfigAdapter.load_legacy_configs(task_config, model_config)
        
        # Create controller
        console.print(f"[blue]Initializing MetaController...[/blue]")
        ctrl = MetaController(task_cfg=unified_config, model_cfg=unified_config)
        
        # Run MACD cycles
        history = []
        console.print(f"[bold cyan]Starting {cycles} MACD cycles...[/bold cyan]")
        
        for c in range(cycles):
            console.print(f"[bold cyan]=== CYCLE {c+1}/{cycles} ===[/bold cyan]")
            
            # Evolution phase
            console.print(f"[blue]Evolution phase...[/blue]")
            best = ctrl.evolve(population=population, top_k=top_k)
            
            # Evaluation phase
            console.print(f"[blue]Evaluation phase...[/blue]")
            val = ctrl.evaluate_once(strategy=best["best_strategy"])
            
            # Distillation phase
            console.print(f"[blue]Distillation phase...[/blue]")
            ctrl.distill_if_improved(val)
            
            # Record history
            history.append({
                "cycle": c + 1,
                "best_score": best["best_score"],
                "validation_score": val,
                "strategy": best["best_strategy"]
            })
            
            console.print(f"[green]Cycle {c+1} completed: score={val:.4f}[/green]")
        
        # Display final results
        if output_format == "table":
            table = Table(title="MACD Cycle Results")
            table.add_column("Cycle", style="cyan")
            table.add_column("Best Score", style="green")
            table.add_column("Validation Score", style="blue")
            
            for h in history:
                table.add_row(
                    str(h["cycle"]),
                    f"{h['best_score']:.4f}",
                    f"{h['validation_score']:.4f}"
                )
            console.print(table)
        else:
            console.print(f"{{'history': {history}}}")
        
        # Save results if requested
        if save_results:
            import json
            with open(results_file, 'w') as f:
                json.dump(history, f, indent=2)
            console.print(f"[green]Results saved to {results_file}[/green]")
        
        console.print(f"[green]✅ All {cycles} cycles completed successfully![/green]")
        
    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found: {e}[/red]")
        raise Exit(1)
    except ConfigurationError as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise Exit(1)
    except ValidationError as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise Exit(1)
    except MACDError as e:
        console.print(f"[red]MACD error: {e}[/red]")
        raise Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise Exit(1)

if __name__ == "__main__":
    app()
