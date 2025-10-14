"""Integration tests for MACD framework."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from macd.core.controller import MetaController
from macd.core.evolution import EvolutionEngine, EvolutionConfig
from macd.core.evaluator import QAEvaluator
from macd.core.trainer import MockTrainer, DistillationStore
from macd.models import MockModel, MockConfig
from macd.config import ConfigManager, MACDConfig, TaskConfig, ModelConfig, ExperimentConfig
from macd.config.schemas import TaskType, ModelBackend


class TestMetaControllerIntegration:
    """Test MetaController integration with all components."""
    
    def test_controller_initialization(self):
        """Test controller initialization with all components."""
        # Create test data
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data.jsonl"
            with open(data_path, 'w') as f:
                json.dump({"prompt": "Is the sky blue?", "answer": "yes"}, f)
                f.write('\n')
                json.dump({"prompt": "Is water dry?", "answer": "no"}, f)
                f.write('\n')
            
            # Create configuration
            task_config = TaskConfig(
                name="test_task",
                type=TaskType.QA,
                data={"eval_path": str(data_path)}
            )
            
            model_config = ModelConfig(
                name="test_model",
                backend=ModelBackend.MOCK,
                params={"device": "cpu"}
            )
            
            # Create controller
            controller = MetaController(task_cfg=task_config, model_cfg=model_config)
            
            assert controller.task_cfg.name == "test_task"
            assert controller.model_cfg.name == "test_model"
            assert isinstance(controller.trainer, MockTrainer)
            assert isinstance(controller.store, DistillationStore)
    
    def test_controller_evaluate_once(self):
        """Test single evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data.jsonl"
            with open(data_path, 'w') as f:
                json.dump({"prompt": "Is the sky blue?", "answer": "yes"}, f)
                f.write('\n')
                json.dump({"prompt": "Is water dry?", "answer": "no"}, f)
                f.write('\n')
            
            task_config = TaskConfig(
                name="test_task",
                type=TaskType.QA,
                data={"eval_path": str(data_path)}
            )
            
            model_config = ModelConfig(
                name="test_model",
                backend=ModelBackend.MOCK,
                params={"device": "cpu"}
            )
            
            controller = MetaController(task_cfg=task_config, model_cfg=model_config)
            
            # Test evaluation
            score = controller.evaluate_once()
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    
    def test_controller_evolve(self):
        """Test evolution process."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data.jsonl"
            with open(data_path, 'w') as f:
                json.dump({"prompt": "Is the sky blue?", "answer": "yes"}, f)
                f.write('\n')
                json.dump({"prompt": "Is water dry?", "answer": "no"}, f)
                f.write('\n')
            
            task_config = TaskConfig(
                name="test_task",
                type=TaskType.QA,
                data={"eval_path": str(data_path)}
            )
            
            model_config = ModelConfig(
                name="test_model",
                backend=ModelBackend.MOCK,
                params={"device": "cpu"}
            )
            
            controller = MetaController(task_cfg=task_config, model_cfg=model_config)
            
            # Test evolution
            result = controller.evolve(population=6, top_k=2)
            
            assert "best_metric" in result
            assert "best_strategy" in result
            assert "elites" in result
            assert "population" in result
            
            assert isinstance(result["best_metric"], float)
            assert isinstance(result["best_strategy"], dict)
            assert len(result["elites"]) == 2
            assert len(result["population"]) == 6
    
    def test_controller_distill_if_improved(self):
        """Test distillation process."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data.jsonl"
            with open(data_path, 'w') as f:
                json.dump({"prompt": "Is the sky blue?", "answer": "yes"}, f)
                f.write('\n')
                json.dump({"prompt": "Is water dry?", "answer": "no"}, f)
                f.write('\n')
            
            task_config = TaskConfig(
                name="test_task",
                type=TaskType.QA,
                data={"eval_path": str(data_path)}
            )
            
            model_config = ModelConfig(
                name="test_model",
                backend=ModelBackend.MOCK,
                params={"device": "cpu"}
            )
            
            controller = MetaController(task_cfg=task_config, model_cfg=model_config)
            
            # Test distillation with improvement
            result = controller.distill_if_improved(best_validation=0.8)
            
            assert "distilled" in result
            assert "curated" in result
            assert "info" in result
            
            if result["distilled"]:
                assert result["curated"] > 0
                assert "updated" in result["info"]
    
    def test_controller_full_cycle(self):
        """Test complete MACD cycle."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data.jsonl"
            with open(data_path, 'w') as f:
                json.dump({"prompt": "Is the sky blue?", "answer": "yes"}, f)
                f.write('\n')
                json.dump({"prompt": "Is water dry?", "answer": "no"}, f)
                f.write('\n')
            
            task_config = TaskConfig(
                name="test_task",
                type=TaskType.QA,
                data={"eval_path": str(data_path)}
            )
            
            model_config = ModelConfig(
                name="test_model",
                backend=ModelBackend.MOCK,
                params={"device": "cpu"}
            )
            
            controller = MetaController(task_cfg=task_config, model_cfg=model_config)
            
            # Run full cycle
            history = []
            for cycle in range(2):
                # Evolve
                evolution_result = controller.evolve(population=4, top_k=2)
                
                # Evaluate best strategy
                validation_score = controller.evaluate_once(
                    strategy=evolution_result["best_strategy"]
                )
                
                # Distill if improved
                distill_result = controller.distill_if_improved(
                    best_validation=validation_score
                )
                
                # Record history
                history.append({
                    "cycle": cycle + 1,
                    "validation": validation_score,
                    "best_metric": evolution_result["best_metric"],
                    "distilled": distill_result["distilled"]
                })
            
            assert len(history) == 2
            for entry in history:
                assert "cycle" in entry
                assert "validation" in entry
                assert "best_metric" in entry
                assert "distilled" in entry
                assert isinstance(entry["validation"], float)
                assert isinstance(entry["best_metric"], float)


class TestConfigManagerIntegration:
    """Test ConfigManager integration with other components."""
    
    def test_config_manager_with_controller(self):
        """Test ConfigManager with MetaController."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            data_path = Path(temp_dir) / "test_data.jsonl"
            with open(data_path, 'w') as f:
                json.dump({"prompt": "Is the sky blue?", "answer": "yes"}, f)
                f.write('\n')
            
            # Create config file
            config_path = Path(temp_dir) / "config.yaml"
            config_data = {
                "experiment": {
                    "name": "integration_test",
                    "seed": 42
                },
                "task": {
                    "name": "test_task",
                    "type": "qa",
                    "data": {"eval_path": str(data_path)}
                },
                "model": {
                    "name": "test_model",
                    "backend": "mock",
                    "params": {"device": "cpu"}
                },
                "evolution": {
                    "population_size": 10,
                    "elite_size": 2
                }
            }
            
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f)
            
            # Load config
            manager = ConfigManager()
            config = manager.load_config(config_path, validate=False)
            
            # Create controller with loaded config
            controller = MetaController(
                task_cfg=config.task,
                model_cfg=config.model
            )
            
            # Test controller functionality
            score = controller.evaluate_once()
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    
    def test_config_manager_environment_overrides(self):
        """Test ConfigManager with environment variable overrides."""
        with patch.dict('os.environ', {
            'MACD_SEED': '123',
            'MACD_POPULATION_SIZE': '20',
            'MACD_MUTATION_RATE': '0.15'
        }):
            manager = ConfigManager()
            config = manager.load_config(validate=False)
            
            assert config.experiment.seed == 123
            assert config.evolution.population_size == 20
            assert config.evolution.mutation_rate == 0.15


class TestEvolutionEngineIntegration:
    """Test EvolutionEngine integration with other components."""
    
    def test_evolution_engine_with_controller(self):
        """Test EvolutionEngine with MetaController."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            data_path = Path(temp_dir) / "test_data.jsonl"
            with open(data_path, 'w') as f:
                json.dump({"prompt": "Is the sky blue?", "answer": "yes"}, f)
                f.write('\n')
                json.dump({"prompt": "Is water dry?", "answer": "no"}, f)
                f.write('\n')
            
            task_config = TaskConfig(
                name="test_task",
                type=TaskType.QA,
                data={"eval_path": str(data_path)}
            )
            
            model_config = ModelConfig(
                name="test_model",
                backend=ModelBackend.MOCK,
                params={"device": "cpu"}
            )
            
            controller = MetaController(task_cfg=task_config, model_cfg=model_config)
            
            # Create evolution engine
            evolution_config = EvolutionConfig(
                population_size=8,
                elite_size=2,
                mutation_rate=0.1,
                crossover_rate=0.8
            )
            
            engine = EvolutionEngine(evolution_config)
            
            # Create initial population
            from macd.core.evolution import random_strategy
            population = [
                (0.5, random_strategy(seed=42)),
                (0.6, random_strategy(seed=123)),
                (0.4, random_strategy(seed=456)),
                (0.7, random_strategy(seed=789))
            ]
            
            # Define fitness function using controller
            def fitness_function(strategy):
                return controller.evaluate_once(strategy=strategy)
            
            # Run evolution
            result = engine.evolve(population, fitness_function)
            
            assert "population" in result
            assert "best_fitness" in result
            assert "diversity" in result
            assert "generation" in result
            assert "converged" in result
            
            assert len(result["population"]) == evolution_config.population_size
            assert engine.generation == 1


class TestEvaluatorIntegration:
    """Test evaluator integration with other components."""
    
    def test_evaluator_with_controller(self):
        """Test evaluator with MetaController."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            data_path = Path(temp_dir) / "test_data.jsonl"
            with open(data_path, 'w') as f:
                json.dump({"prompt": "Is the sky blue?", "answer": "yes"}, f)
                f.write('\n')
                json.dump({"prompt": "Is water dry?", "answer": "no"}, f)
                f.write('\n')
            
            task_config = TaskConfig(
                name="test_task",
                type=TaskType.QA,
                data={"eval_path": str(data_path)}
            )
            
            model_config = ModelConfig(
                name="test_model",
                backend=ModelBackend.MOCK,
                params={"device": "cpu"}
            )
            
            controller = MetaController(task_cfg=task_config, model_cfg=model_config)
            
            # Test evaluator directly
            evaluator = QAEvaluator()
            data = controller._load_data()
            
            # Generate predictions
            model = controller._get_model()
            predictions = []
            references = []
            
            for ex in data:
                pred = model.generate(ex["prompt"], controller.best_strategy)
                predictions.append(pred.text)
                references.append(ex["answer"])
            
            # Evaluate
            result = evaluator.evaluate(predictions, references)
            
            assert isinstance(result.primary_metric, float)
            assert 0.0 <= result.primary_metric <= 1.0
            assert "exact_match" in result.metrics
            assert "f1_score" in result.metrics
            assert result.details["num_examples"] == len(data)


class TestTrainerIntegration:
    """Test trainer integration with other components."""
    
    def test_trainer_with_controller(self):
        """Test trainer with MetaController."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            data_path = Path(temp_dir) / "test_data.jsonl"
            with open(data_path, 'w') as f:
                json.dump({"prompt": "Is the sky blue?", "answer": "yes"}, f)
                f.write('\n')
                json.dump({"prompt": "Is water dry?", "answer": "no"}, f)
                f.write('\n')
            
            task_config = TaskConfig(
                name="test_task",
                type=TaskType.QA,
                data={"eval_path": str(data_path)}
            )
            
            model_config = ModelConfig(
                name="test_model",
                backend=ModelBackend.MOCK,
                params={"device": "cpu"}
            )
            
            controller = MetaController(task_cfg=task_config, model_cfg=model_config)
            
            # Test distillation store
            store = controller.store
            
            # Add examples to store
            store.add("Test prompt 1", "Test prediction 1", "Test reference 1", 0.8)
            store.add("Test prompt 2", "Test prediction 2", "Test reference 2", 0.6)
            store.add("Test prompt 3", "Test prediction 3", "Test reference 3", 0.4)  # Below threshold
            
            assert len(store.curated) == 2  # Only high-reward examples
            
            # Test distillation
            result = controller.trainer.distill(store.curated)
            
            assert "success" in result
            assert "examples_used" in result
            assert result["examples_used"] == 2


class TestEndToEndIntegration:
    """Test end-to-end integration of all components."""
    
    def test_complete_macd_workflow(self):
        """Test complete MACD workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            data_path = Path(temp_dir) / "test_data.jsonl"
            with open(data_path, 'w') as f:
                json.dump({"prompt": "Is the sky blue?", "answer": "yes"}, f)
                f.write('\n')
                json.dump({"prompt": "Is water dry?", "answer": "no"}, f)
                f.write('\n')
                json.dump({"prompt": "Is 2+2 equal to 4?", "answer": "yes"}, f)
                f.write('\n')
            
            # Create configuration
            task_config = TaskConfig(
                name="e2e_test",
                type=TaskType.QA,
                data={"eval_path": str(data_path)}
            )
            
            model_config = ModelConfig(
                name="e2e_model",
                backend=ModelBackend.MOCK,
                params={"device": "cpu"}
            )
            
            experiment_config = ExperimentConfig(
                name="e2e_experiment",
                seed=42
            )
            
            # Create MACD config
            macd_config = MACDConfig(
                experiment=experiment_config,
                task=task_config,
                model=model_config
            )
            
            # Create controller
            controller = MetaController(
                task_cfg=macd_config.task,
                model_cfg=macd_config.model
            )
            
            # Run multiple cycles
            history = []
            for cycle in range(3):
                # Evolution phase
                evolution_result = controller.evolve(population=6, top_k=2)
                
                # Evaluation phase
                validation_score = controller.evaluate_once(
                    strategy=evolution_result["best_strategy"]
                )
                
                # Distillation phase
                distill_result = controller.distill_if_improved(
                    best_validation=validation_score
                )
                
                # Record results
                cycle_result = {
                    "cycle": cycle + 1,
                    "evolution_best": evolution_result["best_metric"],
                    "validation_score": validation_score,
                    "distilled": distill_result["distilled"],
                    "curated_examples": distill_result.get("curated", 0)
                }
                history.append(cycle_result)
            
            # Verify results
            assert len(history) == 3
            
            for cycle_result in history:
                assert "cycle" in cycle_result
                assert "evolution_best" in cycle_result
                assert "validation_score" in cycle_result
                assert "distilled" in cycle_result
                assert "curated_examples" in cycle_result
                
                assert isinstance(cycle_result["evolution_best"], float)
                assert isinstance(cycle_result["validation_score"], float)
                assert isinstance(cycle_result["distilled"], bool)
                assert isinstance(cycle_result["curated_examples"], int)
                
                assert 0.0 <= cycle_result["evolution_best"] <= 1.0
                assert 0.0 <= cycle_result["validation_score"] <= 1.0
                assert cycle_result["curated_examples"] >= 0
            
            # Verify that controller state is maintained
            assert controller.best_validation >= 0.0
            assert isinstance(controller.best_strategy, dict)
            assert len(controller.store.curated) >= 0
