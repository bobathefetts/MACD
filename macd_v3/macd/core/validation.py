"""Input validation utilities for MACD framework."""

import os
import json
from typing import Any, Dict, List, Optional, Union, Callable
from pathlib import Path
import numpy as np

from .exceptions import ValidationError, DataError, FileError, ModelError


class Validator:
    """Input validation utility class."""
    
    @staticmethod
    def validate_not_none(value: Any, name: str) -> None:
        """Validate that value is not None."""
        if value is None:
            raise ValidationError(f"{name} cannot be None")
    
    @staticmethod
    def validate_not_empty(value: Any, name: str) -> None:
        """Validate that value is not empty."""
        if value is None:
            raise ValidationError(f"{name} cannot be None")
        
        if isinstance(value, (str, list, dict, tuple)):
            if len(value) == 0:
                raise ValidationError(f"{name} cannot be empty")
    
    @staticmethod
    def validate_type(value: Any, expected_type: type, name: str) -> None:
        """Validate that value is of expected type."""
        if not isinstance(value, expected_type):
            raise ValidationError(
                f"{name} must be of type {expected_type.__name__}, got {type(value).__name__}"
            )
    
    @staticmethod
    def validate_in_range(
        value: Union[int, float], 
        min_val: Union[int, float], 
        max_val: Union[int, float], 
        name: str
    ) -> None:
        """Validate that value is within range."""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be a number")
        
        if value < min_val or value > max_val:
            raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")
    
    @staticmethod
    def validate_positive(value: Union[int, float], name: str) -> None:
        """Validate that value is positive."""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be a number")
        
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")
    
    @staticmethod
    def validate_non_negative(value: Union[int, float], name: str) -> None:
        """Validate that value is non-negative."""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} must be a number")
        
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")
    
    @staticmethod
    def validate_in_list(value: Any, valid_values: List[Any], name: str) -> None:
        """Validate that value is in list of valid values."""
        if value not in valid_values:
            raise ValidationError(f"{name} must be one of {valid_values}, got {value}")
    
    @staticmethod
    def validate_file_exists(file_path: Union[str, Path], name: str) -> None:
        """Validate that file exists."""
        path = Path(file_path)
        if not path.exists():
            raise FileError(f"{name} file does not exist: {path}")
        
        if not path.is_file():
            raise FileError(f"{name} must be a file, not a directory: {path}")
    
    @staticmethod
    def validate_directory_exists(dir_path: Union[str, Path], name: str) -> None:
        """Validate that directory exists."""
        path = Path(dir_path)
        if not path.exists():
            raise FileError(f"{name} directory does not exist: {path}")
        
        if not path.is_dir():
            raise FileError(f"{name} must be a directory, not a file: {path}")
    
    @staticmethod
    def validate_writable_directory(dir_path: Union[str, Path], name: str) -> None:
        """Validate that directory is writable."""
        path = Path(dir_path)
        
        # Create directory if it doesn't exist
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise FileError(f"Cannot create {name} directory: {path}") from e
        
        # Test write permissions
        try:
            test_file = path / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
        except Exception as e:
            raise FileError(f"{name} directory is not writable: {path}") from e
    
    @staticmethod
    def validate_json_file(file_path: Union[str, Path], name: str) -> Dict[str, Any]:
        """Validate and load JSON file."""
        Validator.validate_file_exists(file_path, name)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                raise ValidationError(f"{name} must contain a JSON object, not {type(data).__name__}")
            
            return data
        
        except json.JSONDecodeError as e:
            raise FileError(f"Invalid JSON in {name}: {file_path}") from e
        except Exception as e:
            raise FileError(f"Error reading {name}: {file_path}") from e
    
    @staticmethod
    def validate_jsonl_file(file_path: Union[str, Path], name: str) -> List[Dict[str, Any]]:
        """Validate and load JSONL file."""
        Validator.validate_file_exists(file_path, name)
        
        try:
            data = []
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        obj = json.loads(line)
                        if not isinstance(obj, dict):
                            raise ValidationError(
                                f"{name} line {line_num} must contain a JSON object"
                            )
                        data.append(obj)
                    except json.JSONDecodeError as e:
                        raise FileError(f"Invalid JSON in {name} line {line_num}: {e}")
            
            if not data:
                raise ValidationError(f"{name} is empty or contains no valid JSON objects")
            
            return data
        
        except Exception as e:
            if isinstance(e, (ValidationError, FileError)):
                raise
            raise FileError(f"Error reading {name}: {file_path}") from e
    
    @staticmethod
    def validate_list_length(
        value: List[Any], 
        expected_length: int, 
        name: str
    ) -> None:
        """Validate that list has expected length."""
        if not isinstance(value, list):
            raise ValidationError(f"{name} must be a list")
        
        if len(value) != expected_length:
            raise ValidationError(
                f"{name} must have length {expected_length}, got {len(value)}"
            )
    
    @staticmethod
    def validate_list_min_length(
        value: List[Any], 
        min_length: int, 
        name: str
    ) -> None:
        """Validate that list has minimum length."""
        if not isinstance(value, list):
            raise ValidationError(f"{name} must be a list")
        
        if len(value) < min_length:
            raise ValidationError(
                f"{name} must have at least {min_length} elements, got {len(value)}"
            )
    
    @staticmethod
    def validate_dict_keys(
        value: Dict[str, Any], 
        required_keys: List[str], 
        name: str
    ) -> None:
        """Validate that dictionary has required keys."""
        if not isinstance(value, dict):
            raise ValidationError(f"{name} must be a dictionary")
        
        missing_keys = [key for key in required_keys if key not in value]
        if missing_keys:
            raise ValidationError(f"{name} missing required keys: {missing_keys}")
    
    @staticmethod
    def validate_dict_optional_keys(
        value: Dict[str, Any], 
        allowed_keys: List[str], 
        name: str
    ) -> None:
        """Validate that dictionary only contains allowed keys."""
        if not isinstance(value, dict):
            raise ValidationError(f"{name} must be a dictionary")
        
        invalid_keys = [key for key in value.keys() if key not in allowed_keys]
        if invalid_keys:
            raise ValidationError(f"{name} contains invalid keys: {invalid_keys}")
    
    @staticmethod
    def validate_numpy_array(
        value: Any, 
        expected_shape: Optional[tuple] = None, 
        expected_dtype: Optional[type] = None,
        name: str = "array"
    ) -> None:
        """Validate numpy array."""
        if not isinstance(value, np.ndarray):
            raise ValidationError(f"{name} must be a numpy array")
        
        if expected_shape is not None and value.shape != expected_shape:
            raise ValidationError(
                f"{name} must have shape {expected_shape}, got {value.shape}"
            )
        
        if expected_dtype is not None and value.dtype != expected_dtype:
            raise ValidationError(
                f"{name} must have dtype {expected_dtype}, got {value.dtype}"
            )
    
    @staticmethod
    def validate_callable(value: Any, name: str) -> None:
        """Validate that value is callable."""
        if not callable(value):
            raise ValidationError(f"{name} must be callable")
    
    @staticmethod
    def validate_environment_variable(var_name: str, required: bool = True) -> Optional[str]:
        """Validate environment variable."""
        value = os.getenv(var_name)
        
        if required and value is None:
            raise ValidationError(f"Required environment variable {var_name} is not set")
        
        return value
    
    @staticmethod
    def validate_api_key(api_key: str, name: str) -> None:
        """Validate API key format."""
        if not api_key:
            raise ValidationError(f"{name} cannot be empty")
        
        if len(api_key) < 10:
            raise ValidationError(f"{name} appears to be too short")
        
        # Basic format validation (can be extended)
        if api_key.startswith('sk-') and len(api_key) < 20:
            raise ValidationError(f"{name} appears to be an invalid OpenAI API key")
    
    @staticmethod
    def validate_model_path(model_path: Union[str, Path], name: str) -> None:
        """Validate model path."""
        path = Path(model_path)
        
        if not path.exists():
            raise ModelError(f"{name} model path does not exist: {path}")
        
        # Check if it's a directory (for local models) or file
        if not (path.is_dir() or path.is_file()):
            raise ModelError(f"{name} model path is neither a file nor directory: {path}")
    
    @staticmethod
    def validate_data_format(
        data: List[Dict[str, Any]], 
        required_fields: List[str], 
        name: str
    ) -> None:
        """Validate data format."""
        Validator.validate_not_empty(data, name)
        Validator.validate_type(data, list, name)
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise DataError(f"{name} item {i} must be a dictionary")
            
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                raise DataError(f"{name} item {i} missing required fields: {missing_fields}")
    
    @staticmethod
    def validate_strategy_format(strategy: Dict[str, Any], name: str = "strategy") -> None:
        """Validate strategy format."""
        Validator.validate_type(strategy, dict, name)
        
        required_sections = ["generation", "retrieval", "training"]
        missing_sections = [section for section in required_sections if section not in strategy]
        if missing_sections:
            raise ValidationError(f"{name} missing required sections: {missing_sections}")
        
        # Validate each section
        for section in required_sections:
            if not isinstance(strategy[section], dict):
                raise ValidationError(f"{name} {section} must be a dictionary")
        
        # Validate prompt_style
        if "prompt_style" not in strategy:
            raise ValidationError(f"{name} missing prompt_style")
        
        valid_styles = ["cot", "concise", "bullet", "json", "detailed", "structured"]
        Validator.validate_in_list(strategy["prompt_style"], valid_styles, f"{name} prompt_style")
    
    @staticmethod
    def validate_metrics_format(metrics: Dict[str, float], name: str = "metrics") -> None:
        """Validate metrics format."""
        Validator.validate_type(metrics, dict, name)
        
        for key, value in metrics.items():
            if not isinstance(key, str):
                raise ValidationError(f"{name} keys must be strings")
            
            if not isinstance(value, (int, float)):
                raise ValidationError(f"{name} values must be numbers")
            
            if not (0.0 <= value <= 1.0):
                raise ValidationError(f"{name} value for {key} must be between 0.0 and 1.0")


def validate_inputs(func: Callable) -> Callable:
    """Decorator to validate function inputs."""
    def wrapper(*args, **kwargs):
        # This is a placeholder for input validation decorator
        # Can be extended to validate specific function inputs
        return func(*args, **kwargs)
    return wrapper


def validate_outputs(func: Callable) -> Callable:
    """Decorator to validate function outputs."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # This is a placeholder for output validation decorator
        # Can be extended to validate specific function outputs
        return result
    return wrapper
