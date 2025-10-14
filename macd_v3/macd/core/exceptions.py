"""Custom exceptions for MACD framework."""

from typing import Optional, Dict, Any


class MACDError(Exception):
    """Base exception for all MACD framework errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class ConfigurationError(MACDError):
    """Exception raised for configuration-related errors."""
    pass


class ModelError(MACDError):
    """Exception raised for model-related errors."""
    pass


class EvaluationError(MACDError):
    """Exception raised for evaluation-related errors."""
    pass


class EvolutionError(MACDError):
    """Exception raised for evolution-related errors."""
    pass


class DistillationError(MACDError):
    """Exception raised for distillation-related errors."""
    pass


class DataError(MACDError):
    """Exception raised for data-related errors."""
    pass


class ValidationError(MACDError):
    """Exception raised for validation-related errors."""
    pass


class TrainingError(MACDError):
    """Exception raised for training-related errors."""
    pass


class ExperimentError(MACDError):
    """Exception raised for experiment-related errors."""
    pass


class LoggingError(MACDError):
    """Exception raised for logging-related errors."""
    pass


class FileError(MACDError):
    """Exception raised for file I/O related errors."""
    pass


class NetworkError(MACDError):
    """Exception raised for network-related errors."""
    pass


class ResourceError(MACDError):
    """Exception raised for resource-related errors (memory, GPU, etc.)."""
    pass


class TimeoutError(MACDError):
    """Exception raised for timeout-related errors."""
    pass


class ConvergenceError(MACDError):
    """Exception raised for convergence-related errors."""
    pass


class CompatibilityError(MACDError):
    """Exception raised for compatibility-related errors."""
    pass
