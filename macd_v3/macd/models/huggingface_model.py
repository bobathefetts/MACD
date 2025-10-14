"""HuggingFace model implementation for MACD framework."""

from typing import Dict, Any, List, Optional
from .base import BaseModel, ModelConfig, GenerationResult

try:
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        GenerationConfig,
        BitsAndBytesConfig
    )
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    GenerationConfig = None
    BitsAndBytesConfig = None


class HuggingFaceConfig(ModelConfig):
    """Configuration for HuggingFace models."""
    model_path: str
    tokenizer_path: Optional[str] = None
    use_quantization: bool = False
    quantization_config: Optional[Dict[str, Any]] = None
    trust_remote_code: bool = False
    torch_dtype: str = "auto"


class HuggingFaceModel(BaseModel):
    """HuggingFace transformers model implementation."""
    
    def __init__(self, config: HuggingFaceConfig):
        super().__init__(config)
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("HuggingFace libraries are not installed. Install with: pip install torch transformers")
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load tokenizer and model."""
        try:
            # Load tokenizer
            tokenizer_path = self.config.tokenizer_path or self.config.model_path
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=self.config.trust_remote_code
            )

            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Determine device strategy
            if self.config.device == "auto":
                # Auto-detect: use GPU if available
                use_gpu = torch.cuda.is_available()
                device_map = "auto" if use_gpu else None
                torch_dtype = torch.float16 if use_gpu else torch.float32
            else:
                # Explicit device specified
                device_map = None
                torch_dtype = torch.float32

            # Prepare model loading kwargs
            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                "device_map": device_map,
            }

            # Handle quantization
            if self.config.use_quantization and self.config.quantization_config:
                quantization_config = BitsAndBytesConfig(**self.config.quantization_config)
                model_kwargs["quantization_config"] = quantization_config

            # Handle torch dtype
            if self.config.torch_dtype != "auto":
                torch_dtype = getattr(torch, self.config.torch_dtype)
            model_kwargs["torch_dtype"] = torch_dtype

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )

            # Move to device if not using device_map
            if self.config.device != "auto" and hasattr(self.model, "to"):
                self.model = self.model.to(self.config.device)

        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model: {e}")
    
    def generate(
        self, 
        prompt: str, 
        strategy: Dict[str, Any],
        **kwargs
    ) -> GenerationResult:
        """Generate text using HuggingFace model."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        params = self.get_strategy_params(strategy)
        
        # Apply prompt style
        formatted_prompt = self._apply_prompt_style(prompt, strategy)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Move to device (only needed if not using device_map="auto")
            if self.config.device != "auto":
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            elif not hasattr(self.model, "hf_device_map"):
                # If device="auto" but model doesn't have device_map, move to GPU if available
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
            
            # Create generation config
            generation_config = GenerationConfig(
                temperature=params["temperature"],
                top_p=params["top_p"],
                max_new_tokens=params["max_tokens"],
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    **kwargs
                )
            
            # Decode output
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return GenerationResult(
                text=generated_text.strip(),
                tokens_used=len(generated_tokens),
                finish_reason="stop",
                metadata={
                    "model_path": self.config.model_path,
                    "input_tokens": input_length,
                    "generated_tokens": len(generated_tokens),
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"HuggingFace generation error: {e}")
    
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
            "cot": f"Question: {prompt}\nLet's think step by step.\nAnswer:",
            "concise": prompt,
            "bullet": f"Please provide a bullet-point answer to: {prompt}\nAnswer:",
            "json": f"Please provide your answer in JSON format for: {prompt}\nAnswer:",
            "detailed": f"Please provide a detailed explanation for: {prompt}\nAnswer:",
        }

        return style_templates.get(style, style_templates["concise"])

    def load_peft_adapter(self, adapter_path: str) -> None:
        """
        Load a PEFT (LoRA) adapter for the model.

        Args:
            adapter_path: Path to the saved PEFT adapter

        Raises:
            ImportError: If PEFT is not installed
            RuntimeError: If loading fails
        """
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError(
                "PEFT library is not installed. Install with: pip install peft"
            )

        if self.model is None:
            raise RuntimeError("Base model not loaded")

        try:
            # Load the PEFT adapter onto the base model
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.eval()  # Set to evaluation mode
        except Exception as e:
            raise RuntimeError(f"Failed to load PEFT adapter from {adapter_path}: {e}")
