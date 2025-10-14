
"""
MACD v3 Command Line Interface

This module provides the command-line interface for the MACD framework,
allowing users to run experiments, evaluations, and evolution cycles
from the command line.

Usage:
    python -m macd.main evaluate --config configs/tasks/qa.yaml
    python -m macd.main evolve --config configs/tasks/qa.yaml --population 50 --top-k 5
    python -m macd.main cycle --config configs/tasks/qa.yaml --cycles 3
"""

from typer import Typer, Option, Exit
from omegaconf import OmegaConf
from pathlib import Path
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional

from .core.controller import MetaController
from .core.utils import seed_everything
from .core.exceptions import MACDError, ConfigurationError, ValidationError
from .config import ConfigManager
from .logging import setup_logger

# Initialize console and app
console = Console()
app = Typer(
    name="macd",
    help="MACD v3 - Meta-Adaptive Context Distillation Framework",
    add_completion=False,
    rich_markup_mode="rich"
)

@app.command()
def evaluate(
    config: str = Option(..., help="Path to unified YAML configuration file."),
    seed: int = Option(42, help="Random number generator seed for reproducibility."),
    output_format: str = Option("json", help="Output format: 'json' or 'table'."),
    verbose: bool = Option(False, help="Enable verbose logging."),
):
    """
    Evaluate a model on a task using the current best strategy.
    
    This command loads the unified configuration file, initializes a MetaController,
    and evaluates the model's performance on the task using the current best strategy.
    
    Args:
        config: Path to YAML file containing complete configuration
        seed: Random seed for reproducibility
        output_format: Format for output display ('json' or 'table')
        verbose: Enable detailed logging output
    """
    try:
        # Setup logging
        if verbose:
            setup_logger(log_level="DEBUG")
        else:
            setup_logger(log_level="INFO")
        
        # Set random seed
        seed_everything(seed)

        # Load configuration
        console.print(f"[blue]Loading configuration...[/blue]")
        config_manager = ConfigManager()
        full_config = config_manager.load_config(config)

        # Create controller with separate task and model configs
        console.print(f"[blue]Initializing MetaController...[/blue]")
        ctrl = MetaController(
            task_cfg=full_config.task,
            model_cfg=full_config.model,
            use_mock_trainer=True  # Default to mock for evaluate command
        )

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
            print({"score": score})
            
        console.print(f"[green]Evaluation completed successfully![/green]")
        
    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found: {e}[/red]")
        raise Exit(1)
    except (ConfigurationError, ValidationError) as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise Exit(1)
    except MACDError as e:
        console.print(f"[red]MACD error: {e}[/red]")
        raise Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise Exit(1)

@app.command()
def evolve(
    config: str = Option(..., help="Path to unified YAML configuration file."),
    population: int = Option(8, help="Population size for evolution."),
    top_k: int = Option(3, help="Number of elite strategies to keep."),
    seed: int = Option(42, help="Random number generator seed for reproducibility."),
    output_format: str = Option("json", help="Output format: 'json' or 'table'."),
    verbose: bool = Option(False, help="Enable verbose logging."),
):
    """
    Run evolution to find better strategies.
    
    This command runs a single evolution cycle to find better strategies
    for the given task and model. It evaluates a population of strategies,
    selects the best ones, and generates new strategies through mutation
    and crossover.
    
    Args:
        config: Path to YAML file containing complete configuration
        population: Size of the population to evolve
        top_k: Number of elite strategies to keep
        seed: Random seed for reproducibility
        output_format: Format for output display ('json' or 'table')
        verbose: Enable detailed logging output
    """
    try:
        # Setup logging
        if verbose:
            setup_logger(log_level="DEBUG")
        else:
            setup_logger(log_level="INFO")
        
        # Set random seed
        seed_everything(seed)
        
        # Load configuration
        console.print(f"[blue]Loading configuration...[/blue]")
        config_manager = ConfigManager()
        full_config = config_manager.load_config(config)

        # Create controller with separate task and model configs
        console.print(f"[blue]Initializing MetaController...[/blue]")
        ctrl = MetaController(
            task_cfg=full_config.task,
            model_cfg=full_config.model,
            use_mock_trainer=True  # Default to mock for evolve command
        )

        # Run evolution
        console.print(f"[blue]Running evolution (population={population}, top_k={top_k})...[/blue]")
        result = ctrl.evolve(population=population, top_k=top_k)
        
        # Display results
        if output_format == "table":
            table = Table(title="Evolution Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Best Metric", f"{result['best_metric']:.4f}")
            table.add_row("Population Size", str(len(result['population'])))
            table.add_row("Elites", str(len(result['elites'])))
            console.print(table)
        else:
            print(result)
            
        console.print(f"[green]Evolution completed successfully![/green]")
        
    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found: {e}[/red]")
        raise Exit(1)
    except (ConfigurationError, ValidationError) as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise Exit(1)
    except MACDError as e:
        console.print(f"[red]MACD error: {e}[/red]")
        raise Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise Exit(1)

@app.command()
def cycle(
    config: str = Option(..., help="Path to unified YAML configuration file."),
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
    Run complete MACD cycles with evolution, evaluation, and distillation.
    
    This command runs multiple complete MACD cycles, each consisting of:
    1. Evolution: Find better strategies through genetic algorithms
    2. Evaluation: Assess the best strategy on validation data
    3. Distillation: Update the model if performance improved
    
    Args:
        config: Path to YAML file containing complete configuration
        cycles: Number of complete MACD cycles to run
        population: Size of the population to evolve
        top_k: Number of elite strategies to keep
        seed: Random seed for reproducibility
        output_format: Format for output display ('json' or 'table')
        verbose: Enable detailed logging output
        save_results: Whether to save results to file
        results_file: File path to save results
    """
    try:
        # Setup logging
        if verbose:
            setup_logger(log_level="DEBUG")
        else:
            setup_logger(log_level="INFO")
        
        # Set random seed
        seed_everything(seed)
        
        # Load configuration
        console.print(f"[blue]Loading configuration...[/blue]")
        config_manager = ConfigManager()
        full_config = config_manager.load_config(config)

        # Create controller with separate task and model configs
        console.print(f"[blue]Initializing MetaController...[/blue]")
        ctrl = MetaController(
            task_cfg=full_config.task,
            model_cfg=full_config.model,
            use_mock_trainer=True  # Use mock trainer by default (user can change config)
        )

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
            distill_result = ctrl.distill_if_improved(best_validation=val)
            
            # Record results
            cycle_result = {
                "cycle": c+1, 
                "validation_score": val, 
                "best_metric": best["best_metric"],
                "distilled": distill_result["distilled"],
                "curated_examples": distill_result.get("curated", 0)
            }
            history.append(cycle_result)
            
            # Display cycle results
            if output_format == "table":
                table = Table(title=f"Cycle {c+1} Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("Validation Score", f"{val:.4f}")
                table.add_row("Best Evolution Metric", f"{best['best_metric']:.4f}")
                table.add_row("Distilled", "Yes" if distill_result["distilled"] else "No")
                table.add_row("Curated Examples", str(distill_result.get("curated", 0)))
                console.print(table)
            else:
                print(cycle_result)
        
        # Display final results
        console.print(f"[bold green]All cycles completed![/bold green]")
        
        if output_format == "table":
            # Summary table
            summary_table = Table(title="MACD Cycle Summary")
            summary_table.add_column("Cycle", style="cyan")
            summary_table.add_column("Validation Score", style="green")
            summary_table.add_column("Best Metric", style="yellow")
            summary_table.add_column("Distilled", style="magenta")
            
            for result in history:
                summary_table.add_row(
                    str(result["cycle"]),
                    f"{result['validation_score']:.4f}",
                    f"{result['best_metric']:.4f}",
                    "Yes" if result["distilled"] else "No"
                )
            console.print(summary_table)
        else:
            print({"history": history})
        
        # Save results if requested
        if save_results:
            import json
            with open(results_file, 'w') as f:
                json.dump({"history": history}, f, indent=2)
            console.print(f"[green]Results saved to {results_file}[/green]")
        
    except FileNotFoundError as e:
        console.print(f"[red]Configuration file not found: {e}[/red]")
        raise Exit(1)
    except (ConfigurationError, ValidationError) as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        raise Exit(1)
    except MACDError as e:
        console.print(f"[red]MACD error: {e}[/red]")
        raise Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise Exit(1)

if __name__ == "__main__":
    app()
