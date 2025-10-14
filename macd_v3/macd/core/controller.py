
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from .evaluator import QAEvaluator, SummarizationEvaluator, EvalResult
from .evolution import random_strategy, mutate, crossover
from .trainer import get_trainer, MockTrainer, DistillationStore, DistillationConfig
from .reward import simple_reward
from .exceptions import (
    MACDError, ModelError, EvaluationError, EvolutionError,
    DistillationError, DataError, ValidationError
)
from .validation import Validator
from ..models import ModelFactory
from ..logging import LoggerMixin
import random

# MockModel has been moved to macd.models.mock_model

@dataclass
class MetaController(LoggerMixin):
    task_cfg: Any
    model_cfg: Any
    use_mock_trainer: bool = True
    trainer: Any = None
    store: DistillationStore = field(default_factory=DistillationStore)
    best_validation: float = 0.0
    best_strategy: Dict[str, Any] = field(default_factory=random_strategy)

    def __post_init__(self):
        """Initialize controller with validation."""
        self._validate_configs()
        self._initialize_components()
        self._initialize_trainer()

    def _validate_configs(self):
        """Validate task and model configurations."""
        try:
            # Validate task config
            if not hasattr(self.task_cfg, 'task') or not hasattr(self.task_cfg.task, 'type'):
                raise ValidationError("Task configuration missing 'task.type'")
            
            # Validate model config
            if not hasattr(self.model_cfg, 'model') or not hasattr(self.model_cfg.model, 'backend'):
                raise ValidationError("Model configuration missing 'model.backend'")
            
            # Validate data path
            if hasattr(self.task_cfg.task, 'data') and hasattr(self.task_cfg.task.data, 'eval_path'):
                Validator.validate_file_exists(self.task_cfg.task.data.eval_path, "Evaluation data")
            
        except Exception as e:
            raise ValidationError(f"Configuration validation failed: {e}") from e
    
    def _initialize_components(self):
        """Initialize controller components."""
        try:
            # Initialize evaluator
            self._evaluator = self._get_evaluator()

            # Initialize model
            self._model = self._get_model()

            # Load and validate data
            self._data = self._load_data()

            self.log_info("MetaController components initialized successfully")

        except Exception as e:
            raise MACDError(f"Component initialization failed: {e}") from e

    def _initialize_trainer(self):
        """Initialize trainer based on configuration."""
        if self.trainer is None:
            try:
                if self.use_mock_trainer:
                    self.trainer = get_trainer(use_mock=True)
                else:
                    # Create distillation config from model config
                    distill_config = self._create_distillation_config()
                    self.trainer = get_trainer(config=distill_config, use_mock=False)
            except ImportError as e:
                self.log_warning(f"Cannot use real trainer: {e}. Falling back to mock.")
                self.trainer = get_trainer(use_mock=True)
                self.use_mock_trainer = True
            except Exception as e:
                raise MACDError(f"Trainer initialization failed: {e}") from e

    def _create_distillation_config(self) -> DistillationConfig:
        """Create distillation configuration from model config."""
        model_params = getattr(self.model_cfg, 'params', {})
        if hasattr(model_params, 'get'):
            base_model_path = model_params.get('model_path', 'gpt2')
        else:
            base_model_path = getattr(model_params, 'model_path', 'gpt2')

        return DistillationConfig(
            base_model_path=base_model_path,
            output_dir=getattr(self.task_cfg, 'output_dir', 'distilled_models'),
            lora_rank=8,
            lora_alpha=16,
            learning_rate=1e-4,
            num_epochs=3,
            batch_size=4
        )

    def _get_evaluator(self):
        """Get evaluator based on task type."""
        try:
            if hasattr(self.task_cfg, 'task'):
                task_type = self.task_cfg.task.type
            else:
                task_type = self.task_cfg.type
                
            if task_type == "qa":
                return QAEvaluator()
            elif task_type == "summarization":
                return SummarizationEvaluator()
            else:
                raise EvaluationError(f"Unsupported task type: {task_type}")
        except Exception as e:
            raise EvaluationError(f"Failed to create evaluator: {e}") from e

    def _load_data(self):
        """Load and validate data."""
        try:
            if hasattr(self.task_cfg, 'task'):
                data_path = self.task_cfg.task.data.eval_path
            else:
                data_path = self.task_cfg.data.eval_path
                
            data = Validator.validate_jsonl_file(data_path, "Evaluation data")
            
            # Validate data format
            required_fields = ["prompt", "answer"]
            Validator.validate_data_format(data, required_fields, "Evaluation data")
            
            self.log_info(f"Loaded {len(data)} examples from {data_path}")
            return data
            
        except Exception as e:
            raise DataError(f"Failed to load data: {e}") from e

    def _get_model(self):
        """Get model instance."""
        try:
            # Use ModelFactory for proper model creation
            if hasattr(self.model_cfg, 'model'):
                model_config_dict = self.model_cfg.model.dict() if hasattr(self.model_cfg.model, 'dict') else self.model_cfg.model
            else:
                model_config_dict = self.model_cfg.dict() if hasattr(self.model_cfg, 'dict') else self.model_cfg
            return ModelFactory.create_model(model_config_dict)

        except Exception as e:
            raise ModelError(f"Failed to create model: {e}") from e

    def _reload_distilled_model(self, distill_result: Dict[str, Any]) -> None:
        """
        Reload model after distillation to use the improved version.

        Args:
            distill_result: Result dictionary from distillation

        Raises:
            ModelError: If model reloading fails
        """
        try:
            # Only reload if distillation was successful and not mock
            if not distill_result.get("success", False):
                return

            if distill_result.get("mock", False):
                self.log_info("Mock trainer used - no model to reload")
                return

            output_dir = distill_result.get("output_dir")
            if not output_dir:
                self.log_warning("No output_dir in distill result - cannot reload model")
                return

            self.log_info(f"Reloading distilled model from {output_dir}")

            # For HuggingFace models with LoRA, we need to reload the PEFT model
            # For other backends, this is a no-op or requires custom implementation
            backend = self.model_cfg.backend if hasattr(self.model_cfg, 'backend') else 'mock'

            if backend == "huggingface":
                # Reload the model with the new LoRA weights
                from ..models.huggingface_model import HuggingFaceModel
                if isinstance(self._model, HuggingFaceModel):
                    self._model.load_peft_adapter(output_dir)
                    self.log_info("Successfully reloaded distilled model")
            else:
                self.log_warning(f"Model reloading not implemented for backend: {backend}")

        except Exception as e:
            raise ModelError(f"Failed to reload distilled model: {e}") from e

    def _generate_predictions(
        self,
        data: List[Dict[str, Any]],
        strategy: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """
        Generate predictions for a dataset using a strategy.

        Args:
            data: List of examples with 'prompt' and 'answer' keys
            strategy: Strategy to use for generation

        Returns:
            Tuple of (predictions, references)
        """
        preds, refs = [], []
        for ex in data:
            try:
                result = self._model.generate(ex["prompt"], strategy)
                preds.append(result.text)
                refs.append(ex["answer"])
            except Exception as e:
                self.log_warning(f"Failed to generate prediction: {e}")
                preds.append("")  # Fallback to empty prediction
                refs.append(ex["answer"])

        return preds, refs

    def evaluate_once(self, strategy: Optional[Dict[str, Any]] = None) -> float:
        """Evaluate model with given strategy."""
        try:
            # Use cached components if available
            data = getattr(self, '_data', self._load_data())
            evaluator = getattr(self, '_evaluator', self._get_evaluator())

            # Validate strategy
            strategy_to_use = strategy or self.best_strategy
            if strategy is not None:
                Validator.validate_strategy_format(strategy, "evaluation strategy")

            # Generate predictions using helper method
            preds, refs = self._generate_predictions(data, strategy_to_use)

            # Evaluate
            if not preds:
                raise EvaluationError("No predictions generated")

            res: EvalResult = evaluator.evaluate(preds, refs)

            self.log_info(f"Evaluation completed: {res.primary_metric:.4f}")
            return res.primary_metric

        except Exception as e:
            self.log_error(f"Evaluation failed: {e}")
            raise EvaluationError(f"Evaluation failed: {e}") from e

    def evolve(self, population: int = 8, top_k: int = 3) -> Dict[str, Any]:
        """Evolve population of strategies."""
        try:
            # Validate inputs
            Validator.validate_positive(population, "population size")
            Validator.validate_positive(top_k, "top_k")
            Validator.validate_in_range(top_k, 1, population, "top_k")
            
            # Use cached components
            data = getattr(self, '_data', self._load_data())
            evaluator = getattr(self, '_evaluator', self._get_evaluator())
            model = getattr(self, '_model', self._get_model())
            
            self.log_info(f"Starting evolution with population={population}, top_k={top_k}")
            
            # Initialize population
            pop = [self.best_strategy] + [random_strategy() for _ in range(max(0, population-1))]
            scored = []
            
            # Evaluate each strategy
            for i, strat in enumerate(pop):
                try:
                    Validator.validate_strategy_format(strat, f"strategy {i}")

                    # Use helper method for prediction generation
                    preds, refs = self._generate_predictions(data, strat)

                    if not preds:
                        raise EvolutionError("No predictions generated for strategy")

                    metric = evaluator.evaluate(preds, refs).primary_metric
                    scored.append((metric, strat))

                except Exception as e:
                    self.log_warning(f"Failed to evaluate strategy {i}: {e}")
                    scored.append((0.0, strat))  # Fallback to zero score
            
            # Sort by fitness
            scored.sort(key=lambda x: x[0], reverse=True)
            elites = [s for _, s in scored[:top_k]]
            
            # Generate children
            children: List[Dict[str, Any]] = []
            rng = random.Random()  # Use unseeded random for stochastic behavior

            while len(elites) + len(children) < population:
                try:
                    # Select parents
                    if len(elites) >= 2:
                        a, b = rng.sample(elites, 2)
                    else:
                        a, b = elites[0], elites[0]
                    
                    # Create child
                    child = mutate(a if rng.random() < 0.5 else b, strength=0.15)
                    Validator.validate_strategy_format(child, "child strategy")
                    children.append(child)
                    
                except Exception as e:
                    self.log_warning(f"Failed to create child: {e}")
                    # Fallback to random strategy
                    children.append(random_strategy())
            
            new_pop = elites + children
            best_metric, best_strategy = scored[0]
            self.best_strategy = best_strategy
            
            result = {
                "best_metric": best_metric,
                "best_strategy": best_strategy,
                "elites": elites,
                "population": new_pop
            }
            
            self.log_info(f"Evolution completed: best_metric={best_metric:.4f}")
            return result
            
        except Exception as e:
            self.log_error(f"Evolution failed: {e}")
            raise EvolutionError(f"Evolution failed: {e}") from e

    def _check_improvement(self, best_validation: float, threshold: float) -> bool:
        """Check if validation score improved sufficiently."""
        Validator.validate_non_negative(best_validation, "best_validation")
        Validator.validate_non_negative(threshold, "threshold")
        return best_validation > self.best_validation + threshold

    def _curate_examples(self, reward_threshold: float = 0.6) -> int:
        """
        Curate high-quality examples for distillation.

        Args:
            reward_threshold: Minimum reward threshold for curation

        Returns:
            Number of examples curated
        """
        data = getattr(self, '_data', self._load_data())
        curated_count = 0

        for ex in data:
            try:
                result = self._model.generate(ex["prompt"], self.best_strategy)
                reward = simple_reward(ex["prompt"], result.text, ex["answer"])

                if self.store.add(ex["prompt"], result.text, ex["answer"], reward=reward, threshold=reward_threshold):
                    curated_count += 1

            except Exception as e:
                self.log_warning(f"Failed to process example for curation: {e}")
                continue

        return curated_count

    def _perform_distillation(self) -> Dict[str, Any]:
        """
        Perform distillation on curated examples.

        Returns:
            Dictionary with distillation results

        Raises:
            DistillationError: If distillation fails
        """
        try:
            info = self.trainer.distill(self.store.curated)

            # Reload the distilled model if successful
            if info.get("success", False) and not info.get("mock", False):
                self._reload_distilled_model(info)

            return info

        except Exception as e:
            self.log_error(f"Distillation failed: {e}")
            raise DistillationError(f"Distillation failed: {e}") from e

    def distill_if_improved(self, best_validation: float, threshold: float = 1e-3):
        """
        Distill model if validation score improved.

        Args:
            best_validation: Current validation score
            threshold: Minimum improvement threshold

        Returns:
            Dictionary with distillation results
        """
        try:
            # Check if improvement is sufficient
            if not self._check_improvement(best_validation, threshold):
                self.log_info(f"No improvement: {best_validation:.4f} <= {self.best_validation:.4f} + {threshold}")
                return {"distilled": False, "reason": "no_improvement"}

            self.log_info(f"Improvement detected: {best_validation:.4f} > {self.best_validation:.4f} + {threshold}")

            # Curate high-quality examples
            curated_count = self._curate_examples(reward_threshold=0.6)

            if curated_count == 0:
                self.log_warning("No examples met quality threshold for distillation")
                return {"distilled": False, "curated": 0, "reason": "no_quality_examples"}

            # Perform distillation
            distill_info = self._perform_distillation()

            # Update best validation score
            self.best_validation = best_validation

            result = {"distilled": True, "curated": curated_count, "info": distill_info}
            self.log_info(f"Distillation completed: {curated_count} examples curated")
            return result

        except Exception as e:
            self.log_error(f"Distillation check failed: {e}")
            raise DistillationError(f"Distillation check failed: {e}") from e
