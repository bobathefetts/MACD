#!/usr/bin/env python3
"""Basic functionality test for MACD v3."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'macd_v3'))

def test_imports():
    """Test that all major imports work."""
    print("Testing imports...")
    
    try:
        from macd.config import ConfigManager
        print("✓ ConfigManager import successful")
    except Exception as e:
        print(f"✗ ConfigManager import failed: {e}")
        return False
    
    try:
        from macd.models import ModelFactory
        print("✓ ModelFactory import successful")
    except Exception as e:
        print(f"✗ ModelFactory import failed: {e}")
        return False
    
    try:
        from macd.core.controller import MetaController
        print("✓ MetaController import successful")
    except Exception as e:
        print(f"✗ MetaController import failed: {e}")
        return False
    
    try:
        from macd.logging import setup_logger
        print("✓ setup_logger import successful")
    except Exception as e:
        print(f"✗ setup_logger import failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from macd.config import ConfigManager
        config_manager = ConfigManager()
        
        # Test loading the QA config
        config = config_manager.load_config("configs/tasks/qa.yaml")
        print("✓ Configuration loading successful")
        print(f"  - Task: {config.task.name}")
        print(f"  - Model: {config.model.name}")
        print(f"  - Backend: {config.model.backend}")
        return True
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        from macd.config import ConfigManager
        from macd.models import ModelFactory
        
        config_manager = ConfigManager()
        config = config_manager.load_config("configs/tasks/qa.yaml")
        
        # Create model using factory
        model = ModelFactory.create_model(config.model.dict())
        print("✓ Model creation successful")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Model name: {model.config.name}")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_controller_creation():
    """Test controller creation."""
    print("\nTesting controller creation...")
    
    try:
        from macd.config import ConfigManager
        from macd.core.controller import MetaController
        
        config_manager = ConfigManager()
        config = config_manager.load_config("configs/tasks/qa.yaml")
        
        # Create controller
        controller = MetaController(task_cfg=config, model_cfg=config)
        print("✓ Controller creation successful")
        print(f"  - Controller type: {type(controller).__name__}")
        return True
    except Exception as e:
        print(f"✗ Controller creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("MACD v3 Basic Functionality Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config_loading,
        test_model_creation,
        test_controller_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MACD v3 is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
