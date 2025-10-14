# MACD v3 Migration Guide

This guide helps you migrate from the old MACD v3 configuration format to the new unified format.

## 🚨 Breaking Changes

### Configuration Format
- **Old**: Separate `task.yaml` and `model.yaml` files
- **New**: Single unified configuration file

### CLI Interface
- **Old**: `--task-config` and `--model-config` parameters
- **New**: Single `--config` parameter

### Model Interface
- **Old**: Direct model instantiation
- **New**: Factory pattern with unified configuration

## 📋 Migration Steps

### Step 1: Update Configuration Files

#### Old Format (Separate Files)

**configs/tasks/qa.yaml** (OLD):
```yaml
task:
  name: toy_qa
  type: qa
  data:
    eval_path: data/qa/dev.jsonl
  evaluator:
    metric: em_f1
search:
  population: 8
  top_k: 3
training:
  distill_threshold: 0.6
```

**configs/models/local.yaml** (OLD):
```yaml
model:
  name: mock
  backend: mock
  params:
    device: cpu
```

#### New Format (Unified File)

**configs/tasks/qa.yaml** (NEW):
```yaml
experiment:
  name: "qa_experiment"
  description: "QA task experiment"
  seed: 42
  output_dir: "outputs"
  log_level: "INFO"
  use_wandb: false
  save_checkpoints: true
  checkpoint_interval: 10
  max_cycles: 10
  early_stopping_patience: 5
  improvement_threshold: 0.001

task:
  name: "qa_task"
  type: "qa"
  data:
    eval_path: "data/qa/dev.jsonl"
  evaluator:
    em_weight: 0.7
    f1_weight: 0.3

model:
  name: "mock_model"
  backend: "mock"
  params:
    device: "cpu"
    seed: 42
    deterministic: true

evolution:
  population_size: 8
  elite_size: 3
  tournament_size: 3
  mutation_rate: 0.1
  crossover_rate: 0.8
  selection_method: "tournament"
  mutation_type: "gaussian"
  adaptive_mutation: true
  diversity_threshold: 0.1
  max_generations: 100
  convergence_threshold: 0.000001
  convergence_window: 10

distillation:
  base_model_path: "microsoft/DialoGPT-small"
  output_dir: "distilled_models"
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.1
  target_modules: ["q_proj", "v_proj"]
  learning_rate: 0.0001
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  warmup_steps: 100
  max_length: 512
  use_quantization: false
  save_steps: 500
  eval_steps: 500
  logging_steps: 100
```

### Step 2: Update CLI Commands

#### Old Commands
```bash
# Old format
python -m macd.main evaluate --task-config configs/tasks/qa.yaml --model-config configs/models/local.yaml
python -m macd.main evolve --task-config configs/tasks/qa.yaml --model-config configs/models/local.yaml
python -m macd.main cycle --task-config configs/tasks/qa.yaml --model-config configs/models/local.yaml --cycles 3
```

#### New Commands
```bash
# New format
python -m macd.main evaluate --config configs/tasks/qa.yaml
python -m macd.main evolve --config configs/tasks/qa.yaml --population 50 --top-k 5
python -m macd.main cycle --config configs/tasks/qa.yaml --cycles 3
```

### Step 3: Update Python Code

#### Old Code
```python
from omegaconf import OmegaConf
from macd.core.controller import MetaController

# Load separate configs
task_cfg = OmegaConf.load("configs/tasks/qa.yaml")
model_cfg = OmegaConf.load("configs/models/local.yaml")

# Create controller
ctrl = MetaController(task_cfg=task_cfg, model_cfg=model_cfg)
```

#### New Code
```python
from macd.config import ConfigManager
from macd.core.controller import MetaController

# Load unified config
config_manager = ConfigManager()
config = config_manager.load_config("configs/tasks/qa.yaml")

# Create controller
ctrl = MetaController(task_cfg=config, model_cfg=config)
```

## 🔄 Backward Compatibility

### Legacy CLI Support

For a smooth transition, we provide legacy CLI commands that support the old format:

```bash
# Legacy commands (with deprecation warnings)
python -m macd.legacy_cli evaluate --task-config configs/tasks/qa.yaml --model-config configs/models/local.yaml
python -m macd.legacy_cli evolve --task-config configs/tasks/qa.yaml --model-config configs/models/local.yaml
python -m macd.legacy_cli cycle --task-config configs/tasks/qa.yaml --model-config configs/models/local.yaml
```

### Legacy Python API

```python
from macd.config.legacy import LegacyConfigAdapter, LegacyMetaController
from omegaconf import OmegaConf

# Convert old configs to new format
task_cfg = OmegaConf.load("configs/tasks/qa.yaml")
model_cfg = OmegaConf.load("configs/models/local.yaml")
unified_config = LegacyConfigAdapter.convert_old_config(task_cfg, model_cfg)

# Or use legacy controller wrapper
ctrl = LegacyMetaController(task_cfg=task_cfg, model_cfg=model_cfg)
```

## 📊 Configuration Mapping

### Task Configuration

| Old Field | New Field | Notes |
|-----------|-----------|-------|
| `task.name` | `task.name` | Direct mapping |
| `task.type` | `task.type` | Direct mapping |
| `task.data.eval_path` | `task.data.eval_path` | Direct mapping |
| `task.evaluator.metric` | `task.evaluator.em_weight` | Split into multiple weights |
| `search.population` | `evolution.population_size` | Renamed |
| `search.top_k` | `evolution.elite_size` | Renamed |
| `training.distill_threshold` | `distillation.distill_threshold` | Moved to distillation section |

### Model Configuration

| Old Field | New Field | Notes |
|-----------|-----------|-------|
| `model.name` | `model.name` | Direct mapping |
| `model.backend` | `model.backend` | Direct mapping |
| `model.params` | `model.params` | Direct mapping |

### New Sections

The new format includes several new configuration sections:

- **`experiment`**: Experiment metadata and settings
- **`evolution`**: Detailed evolution algorithm configuration
- **`distillation`**: Comprehensive distillation settings

## 🛠 Migration Tools

### Automatic Migration Script

```python
from macd.config.legacy import LegacyConfigAdapter
from omegaconf import OmegaConf

# Load old configs
task_cfg = OmegaConf.load("old_task.yaml")
model_cfg = OmegaConf.load("old_model.yaml")

# Convert to new format
new_config = LegacyConfigAdapter.convert_old_config(
    OmegaConf.to_container(task_cfg, resolve=True),
    OmegaConf.to_container(model_cfg, resolve=True)
)

# Save new config
import yaml
with open("new_config.yaml", "w") as f:
    yaml.dump(new_config.dict(), f, default_flow_style=False)
```

### Validation Script

```python
from macd.config import ConfigManager

# Validate new config
config_manager = ConfigManager()
try:
    config = config_manager.load_config("new_config.yaml")
    print("✅ Configuration is valid!")
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

## 🚀 Benefits of New Format

### 1. **Unified Configuration**
- Single file for all settings
- Easier to manage and version control
- Reduced configuration drift

### 2. **Enhanced Validation**
- Pydantic schema validation
- Type checking and constraints
- Better error messages

### 3. **Improved Extensibility**
- Easy to add new configuration options
- Hierarchical configuration structure
- Environment variable support

### 4. **Better Documentation**
- Self-documenting configuration
- Default values and descriptions
- IDE autocompletion support

## 📝 Common Issues and Solutions

### Issue 1: Missing Required Fields
**Error**: `ValidationError: field required`

**Solution**: Add missing required fields to your configuration:
```yaml
experiment:
  name: "my_experiment"  # Required
  description: "My experiment"  # Required
```

### Issue 2: Invalid Field Types
**Error**: `ValidationError: invalid input type`

**Solution**: Check field types in the schema:
```yaml
evolution:
  population_size: 50  # Must be integer
  mutation_rate: 0.1   # Must be float
```

### Issue 3: Out of Range Values
**Error**: `ValidationError: ensure this value is greater than 0`

**Solution**: Ensure values are within valid ranges:
```yaml
evolution:
  population_size: 10  # Must be > 0
  mutation_rate: 0.05  # Must be between 0 and 1
```

## 🔍 Troubleshooting

### Check Configuration Validity
```bash
python -m macd.main evaluate --config your_config.yaml --verbose
```

### Validate with Python
```python
from macd.config import ConfigManager
config_manager = ConfigManager()
config = config_manager.load_config("your_config.yaml")
print("Configuration is valid!")
```

### Use Legacy Mode for Testing
```bash
python -m macd.legacy_cli evaluate --task-config old_task.yaml --model-config old_model.yaml
```

## 📚 Additional Resources

- [Configuration Schema Reference](docs/configuration.md)
- [CLI Reference](docs/cli.md)
- [API Reference](docs/api.md)
- [Examples](examples/)

## 🆘 Getting Help

If you encounter issues during migration:

1. Check the [troubleshooting section](#-troubleshooting)
2. Review the [configuration mapping](#-configuration-mapping)
3. Use the [legacy compatibility layer](#-backward-compatibility)
4. Open an issue on GitHub with your configuration and error message

---

**Note**: The legacy compatibility layer will be removed in a future version. Please migrate to the new format as soon as possible.
