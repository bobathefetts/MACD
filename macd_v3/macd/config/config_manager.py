"""Configuration manager for MACD framework."""

import os
import yaml
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from pydantic import ValidationError

from .schemas import MACDConfig
from loguru import logger


class ConfigManager:
    """Manages configuration loading, validation, and merging."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[MACDConfig] = None
        self._default_config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "experiment": {
                "name": "macd_experiment",
                "description": "MACD experiment",
                "seed": 42,
                "output_dir": "outputs",
                "log_level": "INFO",
                "use_wandb": False,
                "save_checkpoints": True,
                "checkpoint_interval": 10,
                "max_cycles": 10,
                "early_stopping_patience": 5,
                "improvement_threshold": 1e-3,
            },
            "task": {
                "name": "default_task",
                "type": "qa",
                "data": {
                    "eval_path": "data/qa/dev.jsonl"
                },
                "evaluator": {
                    "em_weight": 0.7,
                    "f1_weight": 0.3
                }
            },
            "model": {
                "name": "mock_model",
                "backend": "mock",
                "params": {
                    "device": "auto"  # Auto-detect: GPU if available, else CPU
                }
            },
            "evolution": {
                "population_size": 50,
                "elite_size": 5,
                "tournament_size": 3,
                "mutation_rate": 0.1,
                "crossover_rate": 0.8,
                "selection_method": "tournament",
                "mutation_type": "gaussian",
                "adaptive_mutation": True,
                "diversity_threshold": 0.1,
                "max_generations": 100,
                "convergence_threshold": 1e-6,
                "convergence_window": 10,
            }
        }
    
    def load_config(self, 
                   config_path: Optional[Union[str, Path]] = None,
                   override_config: Optional[Dict[str, Any]] = None,
                   validate: bool = True) -> MACDConfig:
        """
        Load configuration from file and optional overrides.
        
        Args:
            config_path: Path to configuration file
            override_config: Dictionary of configuration overrides
            validate: Whether to validate the configuration
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            ConfigValidationError: If configuration validation fails
        """
        try:
            # Start with default config
            config_dict = self._default_config.copy()
            
            # Load from file if provided
            if config_path:
                config_path = Path(config_path)
                if config_path.exists():
                    config_dict = self._load_from_file(config_path)
                else:
                    logger.warning(f"Configuration file not found: {config_path}")
            
            # Apply overrides
            if override_config:
                config_dict = self._merge_configs(config_dict, override_config)
            
            # Apply environment variable overrides
            config_dict = self._apply_env_overrides(config_dict)
            
            # Create and validate configuration
            self.config = MACDConfig(**config_dict)
            
            if validate:
                self.validate_config()
            
            logger.info(f"Configuration loaded successfully: {self.config.experiment.name}")
            return self.config
            
        except ValidationError as e:
            error_msg = f"Configuration validation failed: {e}"
            logger.error(error_msg)
            raise ConfigValidationError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load configuration: {e}"
            logger.error(error_msg)
            raise ConfigValidationError(error_msg) from e
    
    def _load_from_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            if not isinstance(config_dict, dict):
                raise ValueError("Configuration file must contain a dictionary")
            
            return config_dict
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read configuration file: {e}")
    
    def _merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries recursively."""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        # Common environment variables
        env_mappings = {
            'MACD_SEED': ['experiment', 'seed'],
            'MACD_OUTPUT_DIR': ['experiment', 'output_dir'],
            'MACD_LOG_LEVEL': ['experiment', 'log_level'],
            'MACD_USE_WANDB': ['experiment', 'use_wandb'],
            'MACD_WANDB_PROJECT': ['experiment', 'wandb_project'],
            'MACD_POPULATION_SIZE': ['evolution', 'population_size'],
            'MACD_MUTATION_RATE': ['evolution', 'mutation_rate'],
            'MACD_CROSSOVER_RATE': ['evolution', 'crossover_rate'],
            'OPENAI_API_KEY': ['model', 'params', 'api_key'],
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Navigate to the nested config location
                current = config_dict
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Set the final value with type conversion
                final_key = config_path[-1]
                if env_var in ['MACD_SEED', 'MACD_POPULATION_SIZE']:
                    current[final_key] = int(env_value)
                elif env_var in ['MACD_MUTATION_RATE', 'MACD_CROSSOVER_RATE']:
                    current[final_key] = float(env_value)
                elif env_var in ['MACD_USE_WANDB']:
                    current[final_key] = env_value.lower() in ('true', '1', 'yes', 'on')
                else:
                    current[final_key] = env_value
        
        return config_dict
    
    def validate_config(self) -> None:
        """Validate the loaded configuration."""
        if self.config is None:
            raise ConfigValidationError("No configuration loaded")
        
        try:
            # Validate paths
            self.config.validate_paths()
            
            # Additional custom validations
            self._validate_experiment_config()
            self._validate_task_config()
            self._validate_model_config()
            
            logger.info("Configuration validation passed")
            
        except Exception as e:
            raise ConfigValidationError(f"Configuration validation failed: {e}") from e
    
    def _validate_experiment_config(self) -> None:
        """Validate experiment configuration."""
        exp_config = self.config.experiment
        
        # Validate output directory is writable
        output_dir = Path(exp_config.output_dir)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            test_file = output_dir / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            raise ValueError(f"Output directory is not writable: {output_dir}") from e
        
        # Validate wandb configuration
        if exp_config.use_wandb:
            if not exp_config.wandb_project:
                raise ValueError("Wandb project name is required when use_wandb is True")
    
    def _validate_task_config(self) -> None:
        """Validate task configuration."""
        task_config = self.config.task
        
        # Validate data paths
        for key in ['train_path', 'eval_path', 'test_path']:
            if key in task_config.data:
                path = Path(task_config.data[key])
                if not path.exists():
                    raise ValueError(f"Task data path does not exist: {path}")
    
    def _validate_model_config(self) -> None:
        """Validate model configuration."""
        model_config = self.config.model
        
        # Validate model-specific requirements
        if model_config.backend == "openai":
            if not model_config.params.get('api_key'):
                raise ValueError("OpenAI API key is required for OpenAI backend")
        
        elif model_config.backend == "huggingface":
            model_path = model_config.params.get('model_path')
            if not model_path:
                raise ValueError("Model path is required for HuggingFace backend")
            if not Path(model_path).exists():
                raise ValueError(f"HuggingFace model path does not exist: {model_path}")
    
    def get_config(self) -> MACDConfig:
        """Get the loaded configuration."""
        if self.config is None:
            raise ConfigValidationError("No configuration loaded. Call load_config() first.")
        return self.config
    
    def save_config(self, path: Union[str, Path]) -> None:
        """Save current configuration to file."""
        if self.config is None:
            raise ConfigValidationError("No configuration loaded")
        
        self.config.save(path)
        logger.info(f"Configuration saved to: {path}")
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        if self.config is None:
            raise ConfigValidationError("No configuration loaded")
        
        try:
            # Convert current config to dict
            current_dict = self.config.to_dict()
            
            # Merge updates
            updated_dict = self._merge_configs(current_dict, updates)
            
            # Create new config and validate
            self.config = MACDConfig(**updated_dict)
            self.validate_config()
            
            logger.info("Configuration updated successfully")
            
        except Exception as e:
            raise ConfigValidationError(f"Failed to update configuration: {e}") from e
    
    def get_task_config(self) -> Dict[str, Any]:
        """Get task configuration."""
        return self.get_config().task.dict()
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.get_config().model.dict()
    
    def get_evolution_config(self) -> Dict[str, Any]:
        """Get evolution configuration."""
        return self.get_config().evolution.dict()
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        return self.get_config().experiment.dict()


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass
