"""OpenAI model implementation for MACD framework."""

import os
from typing import Dict, Any, List, Optional
from .base import BaseModel, ModelConfig, GenerationResult

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class OpenAIConfig(ModelConfig):
    """Configuration for OpenAI models."""
    api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    organization: Optional[str] = None
    base_url: Optional[str] = None


class OpenAIModel(BaseModel):
    """OpenAI API model implementation."""
    
    def __init__(self, config: OpenAIConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library is not installed. Install with: pip install openai")
        self.client = self._create_client()
    
    def _create_client(self) -> OpenAI:
        """Create OpenAI client with configuration."""
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via config or OPENAI_API_KEY environment variable")
        
        client_kwargs = {"api_key": api_key}
        if self.config.organization:
            client_kwargs["organization"] = self.config.organization
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
            
        return OpenAI(**client_kwargs)
    
    def generate(
        self, 
        prompt: str, 
        strategy: Dict[str, Any],
        **kwargs
    ) -> GenerationResult:
        """Generate text using OpenAI API."""
        params = self.get_strategy_params(strategy)
        
        # Apply prompt style if specified
        formatted_prompt = self._apply_prompt_style(prompt, strategy)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=params["temperature"],
                top_p=params["top_p"],
                max_tokens=params["max_tokens"],
                **kwargs
            )
            
            choice = response.choices[0]
            return GenerationResult(
                text=choice.message.content,
                tokens_used=response.usage.total_tokens,
                finish_reason=choice.finish_reason,
                metadata={
                    "model": self.config.model_name,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def batch_generate(
        self, 
        prompts: List[str], 
        strategy: Dict[str, Any],
        **kwargs
    ) -> List[GenerationResult]:
        """Generate text for multiple prompts."""
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, strategy, **kwargs)
                results.append(result)
            except Exception as e:
                # Create error result to maintain batch consistency
                results.append(GenerationResult(
                    text="",
                    tokens_used=0,
                    finish_reason="error",
                    metadata={"error": str(e)}
                ))
        return results
    
    def _apply_prompt_style(self, prompt: str, strategy: Dict[str, Any]) -> str:
        """Apply prompt style formatting."""
        style = strategy.get("prompt_style", "concise")
        
        style_templates = {
            "cot": "Let's think step by step.\n\nQuestion: {prompt}\n\nAnswer:",
            "concise": "{prompt}",
            "bullet": "Please provide a bullet-point answer to:\n{prompt}",
            "json": "Please provide your answer in JSON format for:\n{prompt}",
            "detailed": "Please provide a detailed explanation for:\n{prompt}",
        }
        
        template = style_templates.get(style, style_templates["concise"])
        return template.format(prompt=prompt)
