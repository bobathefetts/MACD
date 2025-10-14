# MACD v3 (Meta-Adaptive Context Distillation)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **production-ready, modular, empirically evaluable** framework that implements Meta-Adaptive Context Distillation through evolutionary optimization and knowledge distillation.

## 🚀 Features

- **Real Model Integrations**: OpenAI, HuggingFace, and Mock models with unified interfaces
- **Advanced Evolution**: Multiple genetic algorithms with tournament, roulette, and rank selection
- **Comprehensive Evaluation**: Multiple metrics with statistical analysis and confidence intervals
- **Real Distillation**: LoRA/PEFT-based model fine-tuning on curated high-quality data
- **Robust Configuration**: YAML-based config management with validation and environment overrides
- **Experiment Tracking**: Local and Weights & Biases integration for reproducible experiments
- **Production Ready**: Comprehensive error handling, logging, and input validation
- **Extensive Testing**: 80%+ test coverage with unit, integration, and end-to-end tests

## 📋 Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Migration Guide](#migration-guide)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/your-org/macd_v3.git
cd macd_v3

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Verify Installation

```bash
# Run tests to verify installation
pytest tests/ -v

# Check CLI help
python -m macd.main --help
```

## 🔄 Migration Guide

**⚠️ Important**: MACD v3 has been updated with a new unified configuration format. If you're upgrading from an older version, please see the [Migration Guide](MIGRATION.md) for detailed instructions.

### Quick Migration

**Old format** (deprecated):
```bash
python -m macd.main evaluate --task-config configs/tasks/qa.yaml --model-config configs/models/local.yaml
```

**New format**:
```bash
python -m macd.main evaluate --config configs/tasks/qa.yaml
```

### Legacy Support

For backward compatibility, legacy commands are still available:
```bash
python -m macd.legacy_cli evaluate --task-config configs/tasks/qa.yaml --model-config configs/models/local.yaml
```

See the [Migration Guide](MIGRATION.md) for complete details.

## 🚀 Quickstart

### 1. Basic Evaluation

```bash
# Evaluate a model on QA task
python -m macd.main evaluate \
    --config configs/tasks/qa.yaml \
    --output-format table
```

### 2. Evolution

```bash
# Run evolution to find better strategies
python -m macd.main evolve \
    --task-config configs/tasks/qa.yaml \
    --model-config configs/models/local.yaml \
    --population 20 \
    --top-k 5 \
    --output-format table
```

### 3. Complete MACD Cycle

```bash
# Run multiple MACD cycles with evolution, evaluation, and distillation
python -m macd.main cycle \
    --task-config configs/tasks/qa.yaml \
    --model-config configs/models/local.yaml \
    --cycles 3 \
    --population 20 \
    --top-k 5 \
    --save-results \
    --output-format table
```

### 4. Python API

```python
from macd import ConfigManager, MetaController

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config("config.yaml")

# Create controller
controller = MetaController(
    task_cfg=config.task,
    model_cfg=config.model
)

# Run MACD cycle
for cycle in range(3):
    # Evolution phase
    evolution_result = controller.evolve(population=20, top_k=5)
    
    # Evaluation phase
    validation_score = controller.evaluate_once(evolution_result["best_strategy"])
    
    # Distillation phase
    controller.distill_if_improved(validation_score)
    
    print(f"Cycle {cycle+1}: Score = {validation_score:.4f}")
```

## 🏗 Architecture

### Core Components

```
macd_v3/
├── macd/
│   ├── core/                    # Core framework components
│   │   ├── controller.py        # MetaController - main orchestrator
│   │   ├── evaluator.py         # Evaluation framework with multiple metrics
│   │   ├── evolution.py         # Advanced genetic algorithms
│   │   ├── trainer.py           # Real distillation with LoRA/PEFT
│   │   ├── reward.py            # Reward models for data curation
│   │   ├── exceptions.py        # Custom exception hierarchy
│   │   └── validation.py        # Input validation utilities
│   ├── models/                  # Model implementations
│   │   ├── base.py              # Abstract model interface
│   │   ├── openai_model.py      # OpenAI API integration
│   │   ├── huggingface_model.py # HuggingFace transformers
│   │   ├── mock_model.py        # Mock model for testing
│   │   └── factory.py           # Model factory
│   ├── config/                  # Configuration management
│   │   ├── schemas.py           # Pydantic configuration schemas
│   │   └── config_manager.py    # Configuration loading and validation
│   ├── logging/                 # Logging and experiment tracking
│   │   ├── logger.py            # Structured logging setup
│   │   └── experiment_tracker.py # Local and Wandb tracking
│   └── main.py                  # CLI interface
├── configs/                     # Configuration files
│   ├── tasks/                   # Task configurations
│   ├── models/                  # Model configurations
│   └── defaults.yaml            # Default settings
├── data/                        # Sample datasets
├── tests/                       # Comprehensive test suite
└── requirements.txt             # Dependencies
```

### MACD Loop

The framework implements a complete MACD loop:

1. **Generate**: Create diverse strategies using genetic algorithms
2. **Evaluate**: Assess strategies using task-specific metrics
3. **Select**: Choose top-performing strategies as elites
4. **Evolve**: Generate new strategies through mutation and crossover
5. **Distill**: Fine-tune model on curated high-quality examples

## ⚙️ Configuration

### Configuration Files

The framework uses YAML configuration files with validation:

```yaml
# configs/tasks/qa.yaml
task:
  name: "qa_task"
  type: "qa"
  data:
    eval_path: "data/qa/dev.jsonl"
  evaluator:
    em_weight: 0.7
    f1_weight: 0.3

# configs/models/openai.yaml
model:
  name: "gpt-3.5-turbo"
  backend: "openai"
  params:
    api_key: "${OPENAI_API_KEY}"
    model_name: "gpt-3.5-turbo"
    temperature: 0.7
    max_tokens: 512
```

### Environment Variables

Override configuration with environment variables:

```bash
export MACD_SEED=123
export MACD_POPULATION_SIZE=50
export MACD_MUTATION_RATE=0.1
export OPENAI_API_KEY="your-api-key"
```

## 📖 Usage

### Command Line Interface

#### Evaluate Command

```bash
python -m macd.main evaluate \
    --task-config configs/tasks/qa.yaml \
    --model-config configs/models/openai.yaml \
    --seed 42 \
    --output-format table \
    --verbose
```

#### Evolve Command

```bash
python -m macd.main evolve \
    --task-config configs/tasks/qa.yaml \
    --model-config configs/models/openai.yaml \
    --population 50 \
    --top-k 10 \
    --seed 42 \
    --output-format table
```

#### Cycle Command

```bash
python -m macd.main cycle \
    --task-config configs/tasks/qa.yaml \
    --model-config configs/models/openai.yaml \
    --cycles 5 \
    --population 50 \
    --top-k 10 \
    --save-results \
    --results-file results.json \
    --output-format table
```

### Python API

#### Basic Usage

```python
from macd import ConfigManager, MetaController

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config("config.yaml")

# Create controller
controller = MetaController(
    task_cfg=config.task,
    model_cfg=config.model
)

# Single evaluation
score = controller.evaluate_once()
print(f"Score: {score:.4f}")

# Evolution
result = controller.evolve(population=20, top_k=5)
print(f"Best metric: {result['best_metric']:.4f}")

# Distillation
distill_result = controller.distill_if_improved(score)
print(f"Distilled: {distill_result['distilled']}")
```

#### Advanced Usage with Experiment Tracking

```python
from macd import ConfigManager, MetaController, LocalTracker, setup_logger

# Setup logging and tracking
setup_logger(log_level="INFO", experiment_name="my_experiment")
tracker = LocalTracker("my_experiment", config.to_dict())

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config("config.yaml")

# Create controller
controller = MetaController(
    task_cfg=config.task,
    model_cfg=config.model
)

# Run experiment with tracking
for cycle in range(5):
    # Evolution
    evolution_result = controller.evolve(population=30, top_k=8)
    tracker.log_evolution_results(1, evolution_result)
    
    # Evaluation
    validation_score = controller.evaluate_once(evolution_result["best_strategy"])
    tracker.log_metric("validation_score", validation_score, step=cycle)
    
    # Distillation
    distill_result = controller.distill_if_improved(validation_score)
    tracker.log_distillation_results(distill_result)

tracker.finish()
```

## 📚 API Reference

### Core Classes

#### MetaController

Main orchestrator for the MACD framework.

```python
class MetaController:
    def __init__(self, task_cfg, model_cfg):
        """Initialize controller with task and model configurations."""
    
    def evaluate_once(self, strategy=None):
        """Evaluate model with given strategy."""
    
    def evolve(self, population=8, top_k=3):
        """Evolve population of strategies."""
    
    def distill_if_improved(self, best_validation, threshold=1e-3):
        """Distill model if validation score improved."""
```

#### EvolutionEngine

Advanced genetic algorithm implementation.

```python
class EvolutionEngine:
    def __init__(self, config: EvolutionConfig):
        """Initialize evolution engine with configuration."""
    
    def evolve(self, population, fitness_function):
        """Evolve population for one generation."""
```

#### Evaluators

Task-specific evaluation implementations.

```python
class QAEvaluator(BaseEvaluator):
    def evaluate(self, predictions, references):
        """Evaluate QA predictions with EM and F1 metrics."""

class SummarizationEvaluator(BaseEvaluator):
    def evaluate(self, predictions, references):
        """Evaluate summarization with ROUGE and BLEU metrics."""
```

### Model Interfaces

#### BaseModel

Abstract base class for all model implementations.

```python
class BaseModel(ABC):
    def generate(self, prompt, strategy):
        """Generate text using the model with given strategy."""
    
    def batch_generate(self, prompts, strategy):
        """Generate text for multiple prompts."""
```

#### Model Implementations

- `OpenAIModel`: OpenAI API integration
- `HuggingFaceModel`: HuggingFace transformers
- `MockModel`: Mock model for testing

## 🧪 Examples

### Example 1: QA Task with OpenAI

```python
from macd import ConfigManager, MetaController

# Configuration
config = {
    "task": {
        "name": "qa_task",
        "type": "qa",
        "data": {"eval_path": "data/qa/dev.jsonl"}
    },
    "model": {
        "name": "gpt-3.5-turbo",
        "backend": "openai",
        "params": {
            "api_key": "your-api-key",
            "model_name": "gpt-3.5-turbo"
        }
    }
}

# Run experiment
config_manager = ConfigManager()
macd_config = config_manager.load_config(override_config=config)
controller = MetaController(macd_config.task, macd_config.model)

# Run cycles
for cycle in range(3):
    result = controller.evolve(population=20, top_k=5)
    score = controller.evaluate_once(result["best_strategy"])
    controller.distill_if_improved(score)
    print(f"Cycle {cycle+1}: {score:.4f}")
```

### Example 2: Custom Evaluation Metrics

```python
from macd.core.evaluator import BaseEvaluator, EvalResult

class CustomEvaluator(BaseEvaluator):
    def evaluate(self, predictions, references):
        # Custom evaluation logic
        scores = [self.custom_metric(p, r) for p, r in zip(predictions, references)]
        return EvalResult(
            primary_metric=sum(scores) / len(scores),
            metrics={"custom_metric": sum(scores) / len(scores)},
            details={"scores": scores}
        )
    
    def custom_metric(self, pred, ref):
        # Implement your custom metric
        return 1.0 if pred.lower() == ref.lower() else 0.0
```

### Example 3: Custom Model Integration

```python
from macd.models.base import BaseModel, ModelConfig, GenerationResult

class CustomModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Initialize your custom model
    
    def generate(self, prompt, strategy):
        # Implement generation logic
        text = self.custom_generate(prompt, strategy)
        return GenerationResult(
            text=text,
            tokens_used=len(text.split()),
            finish_reason="stop"
        )
    
    def custom_generate(self, prompt, strategy):
        # Your custom generation logic
        return "Generated response"
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=macd --cov-report=html

# Run specific test categories
pytest tests/test_models.py -v
pytest tests/test_integration.py -v
```

### Test Structure

```
tests/
├── test_models.py          # Model implementation tests
├── test_evolution.py       # Evolution algorithm tests
├── test_evaluator.py       # Evaluation framework tests
├── test_config.py          # Configuration management tests
├── test_controller.py      # Controller integration tests
└── test_integration.py     # End-to-end integration tests
```

### Test Coverage

The framework maintains 80%+ test coverage with:
- Unit tests for individual components
- Integration tests for component interactions
- End-to-end tests for complete workflows
- Mock-based tests for external dependencies

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/macd_v3.git
cd macd_v3

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

```bash
# Format code
black macd/ tests/

# Sort imports
isort macd/ tests/

# Lint code
flake8 macd/ tests/

# Type check
mypy macd/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for the GPT models and API
- HuggingFace for the transformers library
- The evolutionary computation community for genetic algorithm research
- The machine learning community for evaluation metrics and best practices

## 📞 Support

- **Documentation**: [Full Documentation](https://macd-v3.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/your-org/macd_v3/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/macd_v3/discussions)
- **Email**: macd@example.com

---

**MACD v3** - Making language models better through meta-adaptive context distillation.
