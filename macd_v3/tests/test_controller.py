
import pytest
from macd.config import ConfigManager
from macd.core.controller import MetaController

def test_controller_cycle():
    """Test that controller can run a complete cycle."""
    config_manager = ConfigManager()
    config = config_manager.load_config("configs/tasks/qa.yaml")
    
    ctrl = MetaController(task_cfg=config, model_cfg=config)
    before = ctrl.evaluate_once()
    evo = ctrl.evolve(population=6, top_k=2)
    after = ctrl.evaluate_once(strategy=evo["best_strategy"])
    
    # not guaranteed to improve, but should be valid floats in [0,1]
    assert 0.0 <= before <= 1.0
    assert 0.0 <= after <= 1.0
