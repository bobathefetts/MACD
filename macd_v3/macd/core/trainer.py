
import os
import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

try:
    import torch
    import numpy as np
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        TrainingArguments, 
        Trainer as HFTrainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from datasets import Dataset
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    torch = None
    np = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    TrainingArguments = None
    HFTrainer = None
    DataCollatorForLanguageModeling = None
    BitsAndBytesConfig = None
    LoraConfig = None
    get_peft_model = None
    TaskType = None
    PeftModel = None
    Dataset = None


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""
    base_model_path: str
    output_dir: str = "distilled_models"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_length: int = 512
    use_quantization: bool = False
    quantization_config: Optional[Dict[str, Any]] = None
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100


@dataclass
class DistillationStore:
    """Store for curated high-quality examples."""
    curated: List[Dict[str, Any]] = field(default_factory=list)
    min_reward_threshold: float = 0.6
    max_examples: int = 10000

    def add(self, prompt: str, prediction: str, reference: str, reward: float, threshold: Optional[float] = None):
        """Add example to store if it meets quality threshold."""
        import time
        threshold = threshold or self.min_reward_threshold
        if reward >= threshold and len(self.curated) < self.max_examples:
            self.curated.append({
                "prompt": prompt,
                "prediction": prediction,
                "reference": reference,
                "reward": reward,
                "timestamp": time.time()
            })
            return True
        return False

    def get_curated_data(self, max_examples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get curated examples, optionally limited by count."""
        if max_examples is None:
            return self.curated
        return self.curated[:max_examples]

    def clear(self):
        """Clear all curated examples."""
        self.curated.clear()

    def save(self, path: str):
        """Save curated examples to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.curated, f, indent=2)

    def load(self, path: str):
        """Load curated examples from file."""
        if Path(path).exists():
            with open(path, 'r') as f:
                self.curated = json.load(f)


class DistillationTrainer:
    """Real distillation trainer using LoRA/PEFT."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        if not ML_AVAILABLE:
            raise ImportError("ML libraries are not installed. Install with: pip install torch transformers peft datasets")
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self._setup_model()
    
    def _setup_model(self):
        """Setup tokenizer and base model."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            model_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto" if torch.cuda.is_available() else None,
            }
            
            # Add quantization if specified
            if self.config.use_quantization and self.config.quantization_config:
                quantization_config = BitsAndBytesConfig(**self.config.quantization_config)
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_path,
                **model_kwargs
            )
            
            logger.info(f"Loaded base model: {self.config.base_model_path}")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise
    
    def _prepare_dataset(self, examples: List[Dict[str, Any]]) -> Dataset:
        """Prepare dataset for training."""
        if not examples:
            raise ValueError("No examples provided for training")
        
        # Format examples for training
        formatted_examples = []
        for ex in examples:
            # Create training example with prompt and reference
            text = f"{ex['prompt']}\nAnswer: {ex['reference']}"
            formatted_examples.append({"text": text})
        
        # Create dataset
        dataset = Dataset.from_list(formatted_examples)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def distill(self, curated_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform distillation training on curated examples.
        
        Args:
            curated_examples: List of high-quality examples for training
            
        Returns:
            Dictionary with training results
        """
        if not curated_examples:
            logger.warning("No curated examples provided for distillation")
            return {"success": False, "error": "No examples provided"}
        
        try:
            # Prepare dataset
            dataset = self._prepare_dataset(curated_examples)
            logger.info(f"Prepared dataset with {len(dataset)} examples")
            
            # Setup LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
            )
            
            # Apply LoRA to model
            self.peft_model = get_peft_model(self.model, lora_config)
            logger.info("Applied LoRA configuration to model")
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=self.config.output_dir,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="loss",
                greater_is_better=False,
                report_to=None,  # Disable wandb for now
                remove_unused_columns=False,
            )
            
            # Setup data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )
            
            # Create trainer
            trainer = HFTrainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Train
            logger.info("Starting distillation training...")
            train_result = trainer.train()
            
            # Save model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            logger.info(f"Distillation completed. Model saved to {self.config.output_dir}")
            
            return {
                "success": True,
                "train_loss": train_result.training_loss,
                "examples_used": len(curated_examples),
                "output_dir": self.config.output_dir,
                "lora_config": lora_config.to_dict(),
            }
            
        except Exception as e:
            logger.error(f"Distillation training failed: {e}")
            return {"success": False, "error": str(e)}
    
    def load_distilled_model(self, model_path: str):
        """Load a previously distilled model."""
        try:
            self.peft_model = PeftModel.from_pretrained(self.model, model_path)
            logger.info(f"Loaded distilled model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load distilled model: {e}")
            raise


class MockTrainer:
    """Mock trainer for testing and development."""

    def __init__(self):
        self.memory_boost = 0.0
        self.distillation_count = 0

    def distill(self, curated_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Mock distillation that simulates improvement."""
        if not curated_examples:
            return {"success": False, "error": "No examples provided"}

        # Simulate improvement based on number of examples
        gain = 0.01 * len(curated_examples)
        self.memory_boost += gain
        self.distillation_count += 1

        return {
            "success": True,
            "memory_boost": self.memory_boost,
            "examples_used": len(curated_examples),
            "distillation_count": self.distillation_count,
            "simulated_improvement": gain,
            "mock": True  # Flag to indicate this is mock
        }


def get_trainer(config: Optional[DistillationConfig] = None, use_mock: bool = False):
    """
    Factory function to get appropriate trainer instance.

    Args:
        config: Distillation configuration (required for real trainer)
        use_mock: If True, return mock trainer for testing

    Returns:
        Trainer instance (MockTrainer or DistillationTrainer)

    Raises:
        ImportError: If real trainer requested but ML dependencies not installed
        ValueError: If real trainer requested without config
    """
    if use_mock:
        logger.info("Using MockTrainer for testing/development")
        return MockTrainer()

    if not ML_AVAILABLE:
        raise ImportError(
            "ML dependencies not installed. Install with: "
            "pip install torch transformers peft datasets bitsandbytes\n"
            "Or use use_mock=True for testing without real distillation."
        )

    if config is None:
        raise ValueError("DistillationConfig required for real trainer")

    logger.info("Using real DistillationTrainer with LoRA/PEFT")
    return DistillationTrainer(config)


# Backward compatibility - but with deprecation warning
class _DeprecatedTrainer:
    """Deprecated: Use get_trainer() instead."""
    def __new__(cls):
        logger.warning(
            "Direct use of 'Trainer' is deprecated. Use get_trainer(use_mock=True) "
            "or get_trainer(config=...) instead. Falling back to MockTrainer."
        )
        return MockTrainer()

Trainer = _DeprecatedTrainer
