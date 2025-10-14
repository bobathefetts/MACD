#!/usr/bin/env python3
"""
Validation script for MACD v3 setup.
Run this script to verify that the codebase is properly configured.
"""

import sys
import os
from pathlib import Path

# Add the macd_v3 directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def validate_imports():
    """Validate that all imports work correctly."""
    print("🔍 Validating imports...")
    
    try:
        from macd.config import ConfigManager
        print("  ✓ ConfigManager imported successfully")
    except Exception as e:
        print(f"  ✗ ConfigManager import failed: {e}")
        return False
    
    try:
        from macd.models import ModelFactory
        print("  ✓ ModelFactory imported successfully")
    except Exception as e:
        print(f"  ✗ ModelFactory import failed: {e}")
        return False
    
    try:
        from macd.core.controller import MetaController
        print("  ✓ MetaController imported successfully")
    except Exception as e:
        print(f"  ✗ MetaController import failed: {e}")
        return False
    
    try:
        from macd.logging import setup_logger
        print("  ✓ setup_logger imported successfully")
    except Exception as e:
        print(f"  ✗ setup_logger import failed: {e}")
        return False
    
    return True

def validate_config_files():
    """Validate that configuration files exist and are properly formatted."""
    print("\n🔍 Validating configuration files...")
    
    config_files = [
        "configs/tasks/qa.yaml",
        "configs/tasks/summarization.yaml", 
        "configs/models/local.yaml",
        "configs/models/openai.yaml",
        "configs/models/huggingface.yaml"
    ]
    
    for config_file in config_files:
        config_path = current_dir / config_file
        if config_path.exists():
            print(f"  ✓ {config_file} exists")
        else:
            print(f"  ✗ {config_file} missing")
            return False
    
    return True

def validate_data_files():
    """Validate that data files exist."""
    print("\n🔍 Validating data files...")
    
    data_files = [
        "data/qa/dev.jsonl",
        "data/qa/train.jsonl"
    ]
    
    for data_file in data_files:
        data_path = current_dir / data_file
        if data_path.exists():
            print(f"  ✓ {data_file} exists")
        else:
            print(f"  ✗ {data_file} missing")
            return False
    
    return True

def validate_config_loading():
    """Validate that configuration can be loaded."""
    print("\n🔍 Validating configuration loading...")
    
    try:
        from macd.config import ConfigManager
        config_manager = ConfigManager()
        
        # Test loading the QA config
        config = config_manager.load_config("configs/tasks/qa.yaml")
        print("  ✓ Configuration loaded successfully")
        print(f"    - Task: {config.task.name}")
        print(f"    - Model: {config.model.name}")
        print(f"    - Backend: {config.model.backend}")
        return True
    except Exception as e:
        print(f"  ✗ Configuration loading failed: {e}")
        return False

def validate_model_creation():
    """Validate that models can be created."""
    print("\n🔍 Validating model creation...")
    
    try:
        from macd.config import ConfigManager
        from macd.models import ModelFactory
        
        config_manager = ConfigManager()
        config = config_manager.load_config("configs/tasks/qa.yaml")
        
        # Create model using factory
        model = ModelFactory.create_model(config.model.dict())
        print("  ✓ Model created successfully")
        print(f"    - Model type: {type(model).__name__}")
        print(f"    - Model name: {model.config.name}")
        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False

def validate_controller_creation():
    """Validate that controller can be created."""
    print("\n🔍 Validating controller creation...")
    
    try:
        from macd.config import ConfigManager
        from macd.core.controller import MetaController
        
        config_manager = ConfigManager()
        config = config_manager.load_config("configs/tasks/qa.yaml")
        
        # Create controller
        controller = MetaController(task_cfg=config, model_cfg=config)
        print("  ✓ Controller created successfully")
        print(f"    - Controller type: {type(controller).__name__}")
        return True
    except Exception as e:
        print(f"  ✗ Controller creation failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("MACD v3 Setup Validation")
    print("=" * 50)
    
    tests = [
        validate_imports,
        validate_config_files,
        validate_data_files,
        validate_config_loading,
        validate_model_creation,
        validate_controller_creation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"\n❌ Test failed: {test.__name__}")
            break
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        print("✅ MACD v3 is properly configured and ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run CLI help: python -m macd.main --help")
        print("3. Run evaluation: python -m macd.main evaluate --config configs/tasks/qa.yaml")
        return 0
    else:
        print("❌ Some validation tests failed.")
        print("Please fix the issues above before using MACD v3.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
