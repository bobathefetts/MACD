"""Mock model implementation for testing and development."""

import random
import hashlib
from typing import Dict, Any, List
from .base import BaseModel, ModelConfig, GenerationResult


class MockConfig(ModelConfig):
    """Configuration for mock model."""
    seed: int = 42
    deterministic: bool = True


class MockModel(BaseModel):
    """Mock model for testing and development purposes."""
    
    def __init__(self, config: MockConfig):
        super().__init__(config)
        self.rng = random.Random(config.seed)
        self.memory_boost = 0.0  # Simulates model improvement over time
    
    def generate(
        self, 
        prompt: str, 
        strategy: Dict[str, Any],
        **kwargs
    ) -> GenerationResult:
        """Generate mock text based on prompt and strategy."""
        params = self.get_strategy_params(strategy)
        
        # Create deterministic-ish output based on prompt hash
        if self.config.deterministic:
            base = abs(hash((prompt, strategy.get("prompt_style", "")))) % 5
        else:
            base = self.rng.randint(0, 4)
        
        # Define response flavors
        flavors = {
            "cot": "Let's reason step by step. Final answer: ",
            "concise": "",
            "bullet": "- ",
            "json": '{"answer": "',
            "detailed": "Based on the information provided, the answer is: ",
        }
        
        prefix = flavors.get(strategy.get("prompt_style", "concise"), "")
        core_responses = ["yes", "no", "maybe", "unknown", "irrelevant"]
        core = core_responses[base]
        
        # Apply memory boost (simulates model improvement)
        if self.memory_boost > 0 and self.rng.random() < min(0.5, self.memory_boost):
            core = "yes"  # Bias toward positive responses
        
        # Format output
        if prefix == '{"answer": "':
            text = f'{prefix}{core}"}}'
        else:
            text = f"{prefix}{core}"
        
        # Simulate token usage
        tokens_used = len(text.split()) + self.rng.randint(1, 10)
        
        return GenerationResult(
            text=text,
            tokens_used=tokens_used,
            finish_reason="stop",
            metadata={
                "model": "mock",
                "base_response": base,
                "memory_boost": self.memory_boost,
                "temperature": params["temperature"],
            }
        )
    
    def batch_generate(
        self, 
        prompts: List[str], 
        strategy: Dict[str, Any],
        **kwargs
    ) -> List[GenerationResult]:
        """Generate mock text for multiple prompts."""
        return [self.generate(prompt, strategy, **kwargs) for prompt in prompts]
    
    def set_memory_boost(self, boost: float) -> None:
        """Set memory boost for simulating model improvement."""
        self.memory_boost = max(0.0, min(1.0, boost))
