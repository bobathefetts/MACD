"""Base model interface for MACD framework."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pydantic import BaseModel as PydanticBaseModel, Field
from dataclasses import dataclass


class ModelConfig(PydanticBaseModel):
    """Base configuration for all models."""
    name: str = Field(..., description="Model name")
    backend: str = Field(..., description="Model backend type")
    device: str = Field(default="auto", description="Device to run model on (auto=GPU if available, else CPU)")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    
    class Config:
        extra = "allow"


@dataclass
class GenerationResult:
    """Result from model generation."""
    text: str
    tokens_used: int
    finish_reason: str
    metadata: Dict[str, Any] = None


class BaseModel(ABC):
    """Abstract base class for all model implementations."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate model configuration."""
        if not self.config.name:
            raise ValueError("Model name cannot be empty")
        if not self.config.backend:
            raise ValueError("Model backend cannot be empty")
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        strategy: Dict[str, Any],
        **kwargs
    ) -> GenerationResult:
        """
        Generate text using the model with given strategy.
        
        Args:
            prompt: Input prompt
            strategy: Generation strategy parameters
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with generated text and metadata
        """
        pass
    
    @abstractmethod
    def batch_generate(
        self, 
        prompts: List[str], 
        strategy: Dict[str, Any],
        **kwargs
    ) -> List[GenerationResult]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            strategy: Generation strategy parameters
            **kwargs: Additional generation parameters
            
        Returns:
            List of GenerationResult objects
        """
        pass
    
    def get_strategy_params(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate strategy parameters for generation."""
        generation_params = strategy.get("generation", {})
        
        return {
            "temperature": generation_params.get("temperature", self.config.temperature),
            "top_p": generation_params.get("top_p", self.config.top_p),
            "max_tokens": generation_params.get("max_new_tokens", self.config.max_tokens),
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name}, backend={self.config.backend})"
