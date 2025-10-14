"""
MACD v3 (Meta-Adaptive Context Distillation)

A production-ready framework for meta-adaptive context distillation that implements
a credible, modular, and empirically evaluable approach to improving language models
through evolutionary optimization and knowledge distillation.

The framework provides:
- Real model integrations (OpenAI, HuggingFace, Mock)
- Advanced evolution algorithms with multiple genetic operators
- Comprehensive evaluation metrics and statistical analysis
- Real distillation using LoRA/PEFT
- Robust configuration management and validation
- Experiment tracking and logging
- Comprehensive error handling and input validation

Example:
    ```python
    from macd import ConfigManager, MetaController
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config("config.yaml")
    
    # Create controller
    controller = MetaController(
        task_cfg=config.task,
        model_cfg=config.model
    )
    
    # Run MACD cycle
    for cycle in range(3):
        evolution_result = controller.evolve(population=50, top_k=5)
        validation_score = controller.evaluate_once(evolution_result["best_strategy"])
        controller.distill_if_improved(validation_score)
    ```

Author: MACD Development Team
Version: 3.0.0
License: MIT
"""

__version__ = "3.0.0"
__author__ = "MACD Development Team"
__email__ = "macd@example.com"
__license__ = "MIT"

# Core components
from .core.controller import MetaController
from .core.evaluator import QAEvaluator, SummarizationEvaluator, ClassificationEvaluator
from .core.evolution import EvolutionEngine, EvolutionConfig, random_strategy
from .core.trainer import DistillationTrainer, MockTrainer, DistillationStore
from .core.reward import simple_reward
from .core.exceptions import MACDError, ConfigurationError, ModelError, EvaluationError

# Models
from .models import ModelFactory, MockModel, OpenAIModel, HuggingFaceModel

# Configuration
from .config import ConfigManager, MACDConfig

# Legacy compatibility
from .config.legacy import LegacyConfigAdapter, LegacyMetaController

# Logging
from .logging import setup_logger, ExperimentTracker, LocalTracker, WandbTracker

__all__ = [
    # Core
    "MetaController",
    "QAEvaluator", 
    "SummarizationEvaluator",
    "ClassificationEvaluator",
    "EvolutionEngine",
    "EvolutionConfig",
    "random_strategy",
    "DistillationTrainer",
    "MockTrainer", 
    "DistillationStore",
    "simple_reward",
    
    # Exceptions
    "MACDError",
    "ConfigurationError",
    "ModelError", 
    "EvaluationError",
    
    # Models
    "ModelFactory",
    "MockModel",
    "OpenAIModel", 
    "HuggingFaceModel",
    
    # Configuration
    "ConfigManager",
    "MACDConfig",
    
    # Legacy compatibility
    "LegacyConfigAdapter",
    "LegacyMetaController",
    
    # Logging
    "setup_logger",
    "ExperimentTracker",
    "LocalTracker",
    "WandbTracker",
]