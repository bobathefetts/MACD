"""Tests for configuration management."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

from macd.config import (
    ConfigManager,
    TaskConfig,
    ModelConfig,
    EvolutionConfig,
    DistillationConfig,
    ExperimentConfig,
    MACDConfig
)
from macd.config.schemas import (
    TaskType,
    ModelBackend,
    SelectionMethod,
    MutationType
)


class TestTaskConfig:
    """Test TaskConfig validation."""
    
    def test_task_config_creation(self):
        """Test basic task config creation."""
        config = TaskConfig(
            name="test_task",
            type=TaskType.QA,
            data={"eval_path": "data/test.jsonl"},
            evaluator={"em_weight": 0.7}
        )
        
        assert config.name == "test_task"
        assert config.type == TaskType.QA
        assert config.data["eval_path"] == "data/test.jsonl"
        assert config.evaluator["em_weight"] == 0.7
    
    def test_task_config_validation(self):
        """Test task config validation."""
        # Valid config
        config = TaskConfig(
            name="test_task",
            type=TaskType.QA,
            data={"eval_path": "data/test.jsonl"}
        )
        assert config.name == "test_task"
        
        # Invalid task type
        with pytest.raises(ValueError):
            TaskConfig(
                name="test_task",
                type="invalid_type",
                data={"eval_path": "data/test.jsonl"}
            )


class TestModelConfig:
    """Test ModelConfig validation."""
    
    def test_model_config_creation(self):
        """Test basic model config creation."""
        config = ModelConfig(
            name="test_model",
            backend=ModelBackend.MOCK,
            params={"device": "cpu"}
        )
        
        assert config.name == "test_model"
        assert config.backend == ModelBackend.MOCK
        assert config.params["device"] == "cpu"
    
    def test_openai_config_validation(self):
        """Test OpenAI config validation."""
        # Valid OpenAI config
        config = ModelConfig(
            name="openai_model",
            backend=ModelBackend.OPENAI,
            params={
                "api_key": "test_key",
                "model_name": "gpt-3.5-turbo"
            }
        )
        assert config.params["api_key"] == "test_key"
        
        # Invalid OpenAI config (missing api_key)
        with pytest.raises(ValueError, match="OpenAI model requires 'api_key'"):
            ModelConfig(
                name="openai_model",
                backend=ModelBackend.OPENAI,
                params={"model_name": "gpt-3.5-turbo"}
            )
    
    def test_huggingface_config_validation(self):
        """Test HuggingFace config validation."""
        # Valid HuggingFace config
        config = ModelConfig(
            name="hf_model",
            backend=ModelBackend.HUGGINGFACE,
            params={"model_path": "microsoft/DialoGPT-small"}
        )
        assert config.params["model_path"] == "microsoft/DialoGPT-small"
        
        # Invalid HuggingFace config (missing model_path)
        with pytest.raises(ValueError, match="HuggingFace model requires 'model_path'"):
            ModelConfig(
                name="hf_model",
                backend=ModelBackend.HUGGINGFACE,
                params={"device": "cpu"}
            )


class TestEvolutionConfig:
    """Test EvolutionConfig validation."""
    
    def test_evolution_config_creation(self):
        """Test basic evolution config creation."""
        config = EvolutionConfig(
            population_size=50,
            elite_size=5,
            selection_method=SelectionMethod.TOURNAMENT,
            mutation_type=MutationType.GAUSSIAN
        )
        
        assert config.population_size == 50
        assert config.elite_size == 5
        assert config.selection_method == SelectionMethod.TOURNAMENT
        assert config.mutation_type == MutationType.GAUSSIAN
    
    def test_evolution_config_validation(self):
        """Test evolution config validation."""
        # Valid config
        config = EvolutionConfig(
            population_size=50,
            elite_size=5
        )
        assert config.population_size == 50
        
        # Invalid config (elite_size >= population_size)
        with pytest.raises(ValueError, match="Elite size must be less than population size"):
            EvolutionConfig(
                population_size=10,
                elite_size=10
            )
    
    def test_evolution_config_bounds(self):
        """Test evolution config parameter bounds."""
        # Test population size bounds
        with pytest.raises(ValueError):
            EvolutionConfig(population_size=1)  # Too small
        
        with pytest.raises(ValueError):
            EvolutionConfig(population_size=1001)  # Too large
        
        # Test mutation rate bounds
        with pytest.raises(ValueError):
            EvolutionConfig(mutation_rate=-0.1)  # Too small
        
        with pytest.raises(ValueError):
            EvolutionConfig(mutation_rate=1.1)  # Too large


class TestDistillationConfig:
    """Test DistillationConfig validation."""
    
    def test_distillation_config_creation(self):
        """Test basic distillation config creation."""
        config = DistillationConfig(
            base_model_path="microsoft/DialoGPT-small",
            lora_rank=8,
            learning_rate=1e-4
        )
        
        assert config.base_model_path == "microsoft/DialoGPT-small"
        assert config.lora_rank == 8
        assert config.learning_rate == 1e-4
    
    def test_distillation_config_validation(self):
        """Test distillation config validation."""
        # Valid config
        config = DistillationConfig(base_model_path="test/model")
        assert config.base_model_path == "test/model"
        
        # Invalid config (empty base_model_path)
        with pytest.raises(ValueError, match="Base model path cannot be empty"):
            DistillationConfig(base_model_path="")
    
    def test_distillation_config_bounds(self):
        """Test distillation config parameter bounds."""
        # Test LoRA rank bounds
        with pytest.raises(ValueError):
            DistillationConfig(base_model_path="test/model", lora_rank=0)  # Too small
        
        with pytest.raises(ValueError):
            DistillationConfig(base_model_path="test/model", lora_rank=129)  # Too large
        
        # Test learning rate bounds
        with pytest.raises(ValueError):
            DistillationConfig(base_model_path="test/model", learning_rate=1e-7)  # Too small
        
        with pytest.raises(ValueError):
            DistillationConfig(base_model_path="test/model", learning_rate=1e-1)  # Too large


class TestExperimentConfig:
    """Test ExperimentConfig validation."""
    
    def test_experiment_config_creation(self):
        """Test basic experiment config creation."""
        config = ExperimentConfig(
            name="test_experiment",
            seed=42,
            output_dir="outputs",
            log_level="INFO"
        )
        
        assert config.name == "test_experiment"
        assert config.seed == 42
        assert config.output_dir == "outputs"
        assert config.log_level == "INFO"
    
    def test_experiment_config_validation(self):
        """Test experiment config validation."""
        # Valid config
        config = ExperimentConfig(name="test", log_level="DEBUG")
        assert config.log_level == "DEBUG"
        
        # Invalid log level
        with pytest.raises(ValueError, match="Log level must be one of"):
            ExperimentConfig(name="test", log_level="INVALID")
    
    def test_wandb_config_validation(self):
        """Test wandb config validation."""
        # Valid wandb config
        config = ExperimentConfig(
            name="test",
            use_wandb=True,
            wandb_project="test_project"
        )
        assert config.use_wandb is True
        
        # Invalid wandb config (use_wandb=True but no project)
        with pytest.raises(ValueError, match="Wandb project name is required"):
            ExperimentConfig(
                name="test",
                use_wandb=True
            )


class TestMACDConfig:
    """Test MACDConfig integration."""
    
    def test_macd_config_creation(self):
        """Test MACD config creation."""
        task_config = TaskConfig(
            name="test_task",
            type=TaskType.QA,
            data={"eval_path": "data/test.jsonl"}
        )
        
        model_config = ModelConfig(
            name="test_model",
            backend=ModelBackend.MOCK,
            params={"device": "cpu"}
        )
        
        experiment_config = ExperimentConfig(name="test_experiment")
        
        config = MACDConfig(
            experiment=experiment_config,
            task=task_config,
            model=model_config
        )
        
        assert config.experiment.name == "test_experiment"
        assert config.task.name == "test_task"
        assert config.model.name == "test_model"
    
    def test_macd_config_to_dict(self):
        """Test MACD config to dictionary conversion."""
        task_config = TaskConfig(
            name="test_task",
            type=TaskType.QA,
            data={"eval_path": "data/test.jsonl"}
        )
        
        model_config = ModelConfig(
            name="test_model",
            backend=ModelBackend.MOCK,
            params={"device": "cpu"}
        )
        
        experiment_config = ExperimentConfig(name="test_experiment")
        
        config = MACDConfig(
            experiment=experiment_config,
            task=task_config,
            model=model_config
        )
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "experiment" in config_dict
        assert "task" in config_dict
        assert "model" in config_dict
    
    def test_macd_config_save_load(self):
        """Test MACD config save and load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            
            task_config = TaskConfig(
                name="test_task",
                type=TaskType.QA,
                data={"eval_path": "data/test.jsonl"}
            )
            
            model_config = ModelConfig(
                name="test_model",
                backend=ModelBackend.MOCK,
                params={"device": "cpu"}
            )
            
            experiment_config = ExperimentConfig(name="test_experiment")
            
            original_config = MACDConfig(
                experiment=experiment_config,
                task=task_config,
                model=model_config
            )
            
            # Save config
            original_config.save(config_path)
            assert config_path.exists()
            
            # Load config
            loaded_config = MACDConfig.load(config_path)
            assert loaded_config.experiment.name == original_config.experiment.name
            assert loaded_config.task.name == original_config.task.name
            assert loaded_config.model.name == original_config.model.name
    
    def test_macd_config_load_nonexistent_file(self):
        """Test loading config from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            MACDConfig.load("nonexistent_config.yaml")


class TestConfigManager:
    """Test ConfigManager functionality."""
    
    def test_config_manager_initialization(self):
        """Test config manager initialization."""
        manager = ConfigManager()
        assert manager.config is None
        assert manager.config_path is None
        
        # Test with config path
        manager = ConfigManager("test_config.yaml")
        assert manager.config_path == Path("test_config.yaml")
    
    def test_config_manager_load_default(self):
        """Test loading default configuration."""
        manager = ConfigManager()
        config = manager.load_config(validate=False)
        
        assert isinstance(config, MACDConfig)
        assert config.experiment.name == "macd_experiment"
        assert config.task.type == TaskType.QA
        assert config.model.backend == ModelBackend.MOCK
    
    def test_config_manager_load_from_file(self):
        """Test loading configuration from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            
            # Create test config file
            test_config = {
                "experiment": {
                    "name": "file_test",
                    "seed": 123
                },
                "task": {
                    "name": "file_task",
                    "type": "qa",
                    "data": {"eval_path": "data/test.jsonl"}
                },
                "model": {
                    "name": "file_model",
                    "backend": "mock",
                    "params": {"device": "cpu"}
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            # Load config
            manager = ConfigManager()
            config = manager.load_config(config_path, validate=False)
            
            assert config.experiment.name == "file_test"
            assert config.experiment.seed == 123
            assert config.task.name == "file_task"
            assert config.model.name == "file_model"
    
    def test_config_manager_override_config(self):
        """Test configuration overrides."""
        manager = ConfigManager()
        
        overrides = {
            "experiment": {
                "name": "override_test",
                "seed": 456
            },
            "evolution": {
                "population_size": 100
            }
        }
        
        config = manager.load_config(override_config=overrides, validate=False)
        
        assert config.experiment.name == "override_test"
        assert config.experiment.seed == 456
        assert config.evolution.population_size == 100
    
    def test_config_manager_env_overrides(self):
        """Test environment variable overrides."""
        manager = ConfigManager()
        
        with patch.dict('os.environ', {
            'MACD_SEED': '789',
            'MACD_POPULATION_SIZE': '200',
            'MACD_MUTATION_RATE': '0.2'
        }):
            config = manager.load_config(validate=False)
            
            assert config.experiment.seed == 789
            assert config.evolution.population_size == 200
            assert config.evolution.mutation_rate == 0.2
    
    def test_config_manager_validation_error(self):
        """Test configuration validation error."""
        manager = ConfigManager()
        
        # Create invalid config
        invalid_overrides = {
            "evolution": {
                "population_size": 10,
                "elite_size": 15  # Elite size > population size
            }
        }
        
        with pytest.raises(ConfigValidationError):
            manager.load_config(override_config=invalid_overrides, validate=True)
    
    def test_config_manager_save_config(self):
        """Test saving configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "saved_config.yaml"
            
            manager = ConfigManager()
            config = manager.load_config(validate=False)
            
            manager.save_config(config_path)
            assert config_path.exists()
            
            # Verify saved content
            with open(config_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            
            assert saved_config["experiment"]["name"] == "macd_experiment"
    
    def test_config_manager_update_config(self):
        """Test updating configuration."""
        manager = ConfigManager()
        manager.load_config(validate=False)
        
        updates = {
            "experiment": {
                "name": "updated_experiment"
            },
            "evolution": {
                "population_size": 75
            }
        }
        
        manager.update_config(updates)
        
        config = manager.get_config()
        assert config.experiment.name == "updated_experiment"
        assert config.evolution.population_size == 75
    
    def test_config_manager_get_sections(self):
        """Test getting configuration sections."""
        manager = ConfigManager()
        manager.load_config(validate=False)
        
        task_config = manager.get_task_config()
        model_config = manager.get_model_config()
        evolution_config = manager.get_evolution_config()
        experiment_config = manager.get_experiment_config()
        
        assert isinstance(task_config, dict)
        assert isinstance(model_config, dict)
        assert isinstance(evolution_config, dict)
        assert isinstance(experiment_config, dict)
        
        assert "name" in task_config
        assert "name" in model_config
        assert "population_size" in evolution_config
        assert "name" in experiment_config
    
    def test_config_manager_no_config_loaded(self):
        """Test operations when no config is loaded."""
        manager = ConfigManager()
        
        with pytest.raises(ConfigValidationError, match="No configuration loaded"):
            manager.get_config()
        
        with pytest.raises(ConfigValidationError, match="No configuration loaded"):
            manager.save_config("test.yaml")
        
        with pytest.raises(ConfigValidationError, match="No configuration loaded"):
            manager.update_config({})
