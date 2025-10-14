"""Configuration schemas for MACD framework."""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
from pathlib import Path
from enum import Enum


class TaskType(str, Enum):
    """Supported task types."""
    QA = "qa"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    MULTITASK = "multitask"


class ModelBackend(str, Enum):
    """Supported model backends."""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    MOCK = "mock"


class SelectionMethod(str, Enum):
    """Evolution selection methods."""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"


class MutationType(str, Enum):
    """Evolution mutation types."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"


class TaskConfig(BaseModel):
    """Configuration for tasks."""
    name: str = Field(..., description="Task name")
    type: TaskType = Field(..., description="Task type")
    data: Dict[str, Any] = Field(..., description="Data configuration")
    evaluator: Dict[str, Any] = Field(default_factory=dict, description="Evaluator configuration")
    
    @validator('data')
    def validate_data_paths(cls, v):
        """Validate data paths exist."""
        for key in ['train_path', 'eval_path', 'test_path']:
            if key in v:
                path = Path(v[key])
                if not path.exists():
                    raise ValueError(f"Data path does not exist: {path}")
        return v


class ModelConfig(BaseModel):
    """Configuration for models."""
    name: str = Field(..., description="Model name")
    backend: ModelBackend = Field(..., description="Model backend")
    params: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    
    @validator('params')
    def validate_model_params(cls, v, values):
        """Validate model-specific parameters."""
        backend = values.get('backend')
        
        if backend == ModelBackend.OPENAI:
            required_fields = ['api_key', 'model_name']
            for field in required_fields:
                if field not in v:
                    raise ValueError(f"OpenAI model requires '{field}' parameter")
        
        elif backend == ModelBackend.HUGGINGFACE:
            required_fields = ['model_path']
            for field in required_fields:
                if field not in v:
                    raise ValueError(f"HuggingFace model requires '{field}' parameter")
        
        return v


class EvolutionConfig(BaseModel):
    """Configuration for evolution algorithm."""
    population_size: int = Field(default=50, ge=2, le=1000, description="Population size")
    elite_size: int = Field(default=5, ge=1, le=50, description="Number of elite individuals")
    tournament_size: int = Field(default=3, ge=2, le=20, description="Tournament size")
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Mutation rate")
    crossover_rate: float = Field(default=0.8, ge=0.0, le=1.0, description="Crossover rate")
    selection_method: SelectionMethod = Field(default=SelectionMethod.TOURNAMENT, description="Selection method")
    mutation_type: MutationType = Field(default=MutationType.GAUSSIAN, description="Mutation type")
    adaptive_mutation: bool = Field(default=True, description="Use adaptive mutation")
    diversity_threshold: float = Field(default=0.1, ge=0.0, le=1.0, description="Diversity threshold")
    max_generations: int = Field(default=100, ge=1, le=10000, description="Maximum generations")
    convergence_threshold: float = Field(default=1e-6, ge=0.0, le=1.0, description="Convergence threshold")
    convergence_window: int = Field(default=10, ge=1, le=100, description="Convergence window")
    
    @root_validator
    def validate_evolution_params(cls, values):
        """Validate evolution parameters."""
        elite_size = values.get('elite_size', 5)
        population_size = values.get('population_size', 50)
        
        if elite_size >= population_size:
            raise ValueError("Elite size must be less than population size")
        
        return values


class DistillationConfig(BaseModel):
    """Configuration for distillation training."""
    base_model_path: str = Field(..., description="Base model path")
    output_dir: str = Field(default="distilled_models", description="Output directory")
    lora_rank: int = Field(default=8, ge=1, le=128, description="LoRA rank")
    lora_alpha: int = Field(default=16, ge=1, le=256, description="LoRA alpha")
    lora_dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="LoRA dropout")
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"], description="Target modules")
    learning_rate: float = Field(default=1e-4, ge=1e-6, le=1e-2, description="Learning rate")
    num_epochs: int = Field(default=3, ge=1, le=100, description="Number of epochs")
    batch_size: int = Field(default=4, ge=1, le=64, description="Batch size")
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=32, description="Gradient accumulation steps")
    warmup_steps: int = Field(default=100, ge=0, le=10000, description="Warmup steps")
    max_length: int = Field(default=512, ge=1, le=4096, description="Maximum sequence length")
    use_quantization: bool = Field(default=False, description="Use quantization")
    quantization_config: Optional[Dict[str, Any]] = Field(default=None, description="Quantization configuration")
    save_steps: int = Field(default=500, ge=1, le=10000, description="Save steps")
    eval_steps: int = Field(default=500, ge=1, le=10000, description="Evaluation steps")
    logging_steps: int = Field(default=100, ge=1, le=1000, description="Logging steps")
    
    @validator('base_model_path')
    def validate_base_model_path(cls, v):
        """Validate base model path."""
        if not v:
            raise ValueError("Base model path cannot be empty")
        return v


class ExperimentConfig(BaseModel):
    """Configuration for experiments."""
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(default=None, description="Experiment description")
    seed: int = Field(default=42, ge=0, le=2**32-1, description="Random seed")
    output_dir: str = Field(default="outputs", description="Output directory")
    log_level: str = Field(default="INFO", description="Logging level")
    use_wandb: bool = Field(default=False, description="Use Weights & Biases")
    wandb_project: Optional[str] = Field(default=None, description="Wandb project name")
    wandb_entity: Optional[str] = Field(default=None, description="Wandb entity")
    save_checkpoints: bool = Field(default=True, description="Save model checkpoints")
    checkpoint_interval: int = Field(default=10, ge=1, le=1000, description="Checkpoint interval")
    max_cycles: int = Field(default=10, ge=1, le=1000, description="Maximum MACD cycles")
    early_stopping_patience: int = Field(default=5, ge=1, le=50, description="Early stopping patience")
    improvement_threshold: float = Field(default=1e-3, ge=0.0, le=1.0, description="Improvement threshold")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
    
    @root_validator
    def validate_wandb_config(cls, values):
        """Validate wandb configuration."""
        use_wandb = values.get('use_wandb', False)
        wandb_project = values.get('wandb_project')
        
        if use_wandb and not wandb_project:
            raise ValueError("Wandb project name is required when use_wandb is True")
        
        return values


class MACDConfig(BaseModel):
    """Main MACD configuration."""
    experiment: ExperimentConfig = Field(..., description="Experiment configuration")
    task: TaskConfig = Field(..., description="Task configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig, description="Evolution configuration")
    distillation: Optional[DistillationConfig] = Field(default=None, description="Distillation configuration")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.dict()
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to file."""
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MACDConfig':
        """Load configuration from file."""
        import yaml
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    def validate_paths(self) -> None:
        """Validate all file paths in configuration."""
        # Validate task data paths
        for key in ['train_path', 'eval_path', 'test_path']:
            if key in self.task.data:
                path = Path(self.task.data[key])
                if not path.exists():
                    raise ValueError(f"Task data path does not exist: {path}")
        
        # Validate model paths
        if self.model.backend == ModelBackend.HUGGINGFACE:
            model_path = Path(self.model.params.get('model_path', ''))
            if not model_path.exists():
                raise ValueError(f"Model path does not exist: {model_path}")
        
        # Validate distillation paths
        if self.distillation:
            base_model_path = Path(self.distillation.base_model_path)
            if not base_model_path.exists():
                raise ValueError(f"Distillation base model path does not exist: {base_model_path}")
        
        # Validate output directory
        output_dir = Path(self.experiment.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
