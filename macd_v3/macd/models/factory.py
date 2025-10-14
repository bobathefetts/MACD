"""Model factory for creating model instances."""

from typing import Dict, Any, Type
from .base import BaseModel, ModelConfig
from .openai_model import OpenAIModel, OpenAIConfig
from .huggingface_model import HuggingFaceModel, HuggingFaceConfig
from .mock_model import MockModel, MockConfig


class ModelFactory:
    """Factory for creating model instances based on configuration."""
    
    _model_registry: Dict[str, Type[BaseModel]] = {
        "openai": OpenAIModel,
        "huggingface": HuggingFaceModel,
        "mock": MockModel,
    }
    
    _config_registry: Dict[str, Type[ModelConfig]] = {
        "openai": OpenAIConfig,
        "huggingface": HuggingFaceConfig,
        "mock": MockConfig,
    }
    
    @classmethod
    def create_model(cls, config_dict: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance from configuration dictionary.
        
        Args:
            config_dict: Model configuration dictionary
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If backend type is not supported
        """
        backend = config_dict.get("backend", "mock")
        
        if backend not in cls._model_registry:
            raise ValueError(f"Unsupported model backend: {backend}")
        
        # Get config class and create config instance
        config_class = cls._config_registry[backend]
        config = config_class(**config_dict)
        
        # Get model class and create model instance
        model_class = cls._model_registry[backend]
        return model_class(config)
    
    @classmethod
    def register_model(cls, backend: str, model_class: Type[BaseModel], config_class: Type[ModelConfig]) -> None:
        """
        Register a new model type.
        
        Args:
            backend: Backend identifier
            model_class: Model implementation class
            config_class: Configuration class
        """
        cls._model_registry[backend] = model_class
        cls._config_registry[backend] = config_class
    
    @classmethod
    def list_backends(cls) -> list[str]:
        """List all available model backends."""
        return list(cls._model_registry.keys())
