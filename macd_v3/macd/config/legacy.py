"""Legacy configuration support for backward compatibility."""

import warnings
from typing import Dict, Any, Union
from pathlib import Path
from omegaconf import OmegaConf

from .config_manager import ConfigManager
from .schemas import MACDConfig


class LegacyConfigAdapter:
    """Adapter for legacy configuration formats."""
    
    @staticmethod
    def convert_old_config(task_config: Dict[str, Any], model_config: Dict[str, Any]) -> MACDConfig:
        """
        Convert old separate task and model configs to new unified format.
        
        Args:
            task_config: Old task configuration dictionary
            model_config: Old model configuration dictionary
            
        Returns:
            New unified MACDConfig object
        """
        warnings.warn(
            "Using legacy configuration format. Please update to the new unified format. "
            "See migration guide for details.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Extract task information
        task_info = task_config.get("task", {})
        task_name = task_info.get("name", "legacy_task")
        task_type = task_info.get("type", "qa")
        
        # Extract data information
        data_info = task_info.get("data", {})
        eval_path = data_info.get("eval_path", "data/qa/dev.jsonl")
        
        # Extract evaluator information
        evaluator_info = task_info.get("evaluator", {})
        
        # Extract model information
        model_info = model_config.get("model", {})
        model_name = model_info.get("name", "legacy_model")
        model_backend = model_info.get("backend", "mock")
        model_params = model_info.get("params", {})
        
        # Extract search/evolution information
        search_info = task_config.get("search", {})
        population_size = search_info.get("population", 8)
        elite_size = search_info.get("top_k", 3)
        
        # Extract training information
        training_info = task_config.get("training", {})
        distill_threshold = training_info.get("distill_threshold", 0.6)
        
        # Create new unified config
        new_config_dict = {
            "experiment": {
                "name": f"{task_name}_experiment",
                "description": f"Legacy experiment for {task_name}",
                "seed": 42,
                "output_dir": "outputs",
                "log_level": "INFO",
                "use_wandb": False,
                "save_checkpoints": True,
                "checkpoint_interval": 10,
                "max_cycles": 10,
                "early_stopping_patience": 5,
                "improvement_threshold": 0.001,
            },
            "task": {
                "name": task_name,
                "type": task_type,
                "data": {
                    "eval_path": eval_path
                },
                "evaluator": {
                    "em_weight": evaluator_info.get("em_weight", 0.7),
                    "f1_weight": evaluator_info.get("f1_weight", 0.3),
                    "rouge_weight": evaluator_info.get("rouge_weight", 0.4),
                    "bleu_weight": evaluator_info.get("bleu_weight", 0.3),
                }
            },
            "model": {
                "name": model_name,
                "backend": model_backend,
                "params": model_params
            },
            "evolution": {
                "population_size": population_size,
                "elite_size": elite_size,
                "tournament_size": 3,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "selection_method": "tournament",
                "mutation_type": "gaussian",
                "adaptive_mutation": True,
                "diversity_threshold": 0.1,
                "max_generations": 100,
                "convergence_threshold": 0.000001,
                "convergence_window": 10,
            },
            "distillation": {
                "base_model_path": "microsoft/DialoGPT-small",
                "output_dir": "distilled_models",
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"],
                "learning_rate": 0.0001,
                "num_epochs": 3,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 100,
                "max_length": 512,
                "use_quantization": False,
                "save_steps": 500,
                "eval_steps": 500,
                "logging_steps": 100,
                "distill_threshold": distill_threshold,
                "max_curated_examples": 10000,
            }
        }
        
        return MACDConfig(**new_config_dict)
    
    @staticmethod
    def load_legacy_configs(task_config_path: Union[str, Path], model_config_path: Union[str, Path]) -> MACDConfig:
        """
        Load and convert legacy configuration files.
        
        Args:
            task_config_path: Path to old task configuration file
            model_config_path: Path to old model configuration file
            
        Returns:
            New unified MACDConfig object
        """
        # Load old configs
        task_config = OmegaConf.load(task_config_path)
        model_config = OmegaConf.load(model_config_path)
        
        # Convert to new format
        return LegacyConfigAdapter.convert_old_config(
            OmegaConf.to_container(task_config, resolve=True),
            OmegaConf.to_container(model_config, resolve=True)
        )


def create_legacy_compatibility_wrapper():
    """Create a compatibility wrapper for the old MetaController interface."""
    
    class LegacyMetaController:
        """Legacy MetaController wrapper for backward compatibility."""
        
        def __init__(self, task_cfg, model_cfg):
            warnings.warn(
                "Using legacy MetaController interface. Please update to use the new unified config format. "
                "See migration guide for details.",
                DeprecationWarning,
                stacklevel=2
            )
            
            # Convert old configs to new format
            if hasattr(task_cfg, 'to_container'):
                task_dict = OmegaConf.to_container(task_cfg, resolve=True)
            else:
                task_dict = task_cfg
                
            if hasattr(model_cfg, 'to_container'):
                model_dict = OmegaConf.to_container(model_cfg, resolve=True)
            else:
                model_dict = model_cfg
            
            # Convert to new unified config
            unified_config = LegacyConfigAdapter.convert_old_config(task_dict, model_dict)
            
            # Import and create new controller
            from ..core.controller import MetaController
            self._controller = MetaController(task_cfg=unified_config, model_cfg=unified_config)
        
        def evaluate_once(self, strategy=None):
            """Evaluate model with given strategy."""
            return self._controller.evaluate_once(strategy)
        
        def evolve(self, population=8, top_k=3):
            """Evolve population of strategies."""
            return self._controller.evolve(population=population, top_k=top_k)
        
        def distill_if_improved(self, validation_score):
            """Distill model if performance improved."""
            return self._controller.distill_if_improved(validation_score)
        
        def cycle(self, population=8, top_k=3):
            """Run a complete MACD cycle."""
            return self._controller.cycle(population=population, top_k=top_k)
    
    return LegacyMetaController


# Create the legacy controller class
LegacyMetaController = create_legacy_compatibility_wrapper()
