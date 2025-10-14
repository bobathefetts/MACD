"""Logging setup for MACD framework."""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
from datetime import datetime


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    experiment_name: Optional[str] = None,
    use_console: bool = True,
    log_format: Optional[str] = None
) -> None:
    """
    Setup logging configuration for MACD framework.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Custom log file path
        log_dir: Directory for log files
        experiment_name: Name of the experiment for log file naming
        use_console: Whether to log to console
        log_format: Custom log format string
    """
    # Remove default handler
    logger.remove()
    
    # Default log format
    if log_format is None:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Setup console logging
    if use_console:
        logger.add(
            sys.stderr,
            format=log_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
    
    # Setup file logging
    if log_file is None:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Generate log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            log_file = log_path / f"{experiment_name}_{timestamp}.log"
        else:
            log_file = log_path / f"macd_{timestamp}.log"
    
    # Add file handler
    logger.add(
        str(log_file),
        format=log_format,
        level=log_level,
        rotation="100 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True
    )
    
    # Log setup completion
    logger.info(f"Logging setup complete. Log level: {log_level}")
    logger.info(f"Log file: {log_file}")


def get_logger(name: Optional[str] = None) -> Any:
    """
    Get logger instance.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self):
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)
    
    def log_info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None, **kwargs):
        """Log metric value."""
        if step is not None:
            self.logger.info(f"METRIC: {name}={value} (step={step})", **kwargs)
        else:
            self.logger.info(f"METRIC: {name}={value}", **kwargs)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration."""
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_experiment_start(self, experiment_name: str, config: Dict[str, Any]):
        """Log experiment start."""
        self.logger.info("=" * 80)
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.logger.info("=" * 80)
        self.log_config(config)
    
    def log_experiment_end(self, experiment_name: str, results: Dict[str, Any]):
        """Log experiment end."""
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment completed: {experiment_name}")
        self.logger.info("Results:")
        for key, value in results.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 80)
    
    def log_cycle_start(self, cycle: int, total_cycles: int):
        """Log cycle start."""
        self.logger.info(f"Starting cycle {cycle}/{total_cycles}")
    
    def log_cycle_end(self, cycle: int, results: Dict[str, Any]):
        """Log cycle end."""
        self.logger.info(f"Cycle {cycle} completed")
        for key, value in results.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_evolution_start(self, generation: int, population_size: int):
        """Log evolution start."""
        self.logger.info(f"Starting evolution generation {generation} (population: {population_size})")
    
    def log_evolution_end(self, generation: int, best_fitness: float, diversity: float):
        """Log evolution end."""
        self.logger.info(f"Evolution generation {generation} completed")
        self.logger.info(f"  Best fitness: {best_fitness:.4f}")
        self.logger.info(f"  Population diversity: {diversity:.4f}")
    
    def log_distillation_start(self, num_examples: int):
        """Log distillation start."""
        self.logger.info(f"Starting distillation with {num_examples} examples")
    
    def log_distillation_end(self, results: Dict[str, Any]):
        """Log distillation end."""
        self.logger.info("Distillation completed")
        for key, value in results.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_evaluation_start(self, num_examples: int):
        """Log evaluation start."""
        self.logger.info(f"Starting evaluation with {num_examples} examples")
    
    def log_evaluation_end(self, results: Dict[str, Any]):
        """Log evaluation end."""
        self.logger.info("Evaluation completed")
        for key, value in results.items():
            self.logger.info(f"  {key}: {value}")
