"""Tests for model implementations."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from macd.models import (
    MockModel, MockConfig,
    OpenAIModel, OpenAIConfig,
    HuggingFaceModel, HuggingFaceConfig,
    ModelFactory
)
from macd.models.base import GenerationResult


class TestMockModel:
    """Test MockModel implementation."""
    
    def test_mock_model_initialization(self):
        """Test mock model initialization."""
        config = MockConfig(name="test_mock", backend="mock", seed=42)
        model = MockModel(config)
        
        assert model.config.name == "test_mock"
        assert model.config.backend == "mock"
        assert model.config.seed == 42
    
    def test_mock_model_generate(self):
        """Test mock model generation."""
        config = MockConfig(name="test_mock", backend="mock", deterministic=True)
        model = MockModel(config)
        
        strategy = {"prompt_style": "concise"}
        result = model.generate("Test prompt", strategy)
        
        assert isinstance(result, GenerationResult)
        assert isinstance(result.text, str)
        assert result.tokens_used > 0
        assert result.finish_reason == "stop"
        assert "model" in result.metadata
    
    def test_mock_model_batch_generate(self):
        """Test mock model batch generation."""
        config = MockConfig(name="test_mock", backend="mock")
        model = MockModel(config)
        
        prompts = ["Test prompt 1", "Test prompt 2"]
        strategy = {"prompt_style": "cot"}
        results = model.batch_generate(prompts, strategy)
        
        assert len(results) == 2
        for result in results:
            assert isinstance(result, GenerationResult)
            assert "Let's reason step by step" in result.text
    
    def test_mock_model_memory_boost(self):
        """Test mock model memory boost functionality."""
        config = MockConfig(name="test_mock", backend="mock")
        model = MockModel(config)
        
        # Set memory boost
        model.set_memory_boost(0.8)
        assert model.memory_boost == 0.8
        
        # Test generation with memory boost
        strategy = {"prompt_style": "concise"}
        result = model.generate("Test prompt", strategy)
        
        # Should have memory boost in metadata
        assert "memory_boost" in result.metadata
        assert result.metadata["memory_boost"] == 0.8


class TestOpenAIModel:
    """Test OpenAIModel implementation."""
    
    def test_openai_config_validation(self):
        """Test OpenAI config validation."""
        # Valid config
        config = OpenAIConfig(
            name="test_openai",
            backend="openai",
            api_key="test_key",
            model_name="gpt-3.5-turbo"
        )
        assert config.api_key == "test_key"
        assert config.model_name == "gpt-3.5-turbo"
        
        # Invalid config (missing api_key)
        with pytest.raises(ValueError):
            OpenAIConfig(
                name="test_openai",
                backend="openai",
                model_name="gpt-3.5-turbo"
            )
    
    @patch('macd.models.openai_model.OpenAI')
    def test_openai_model_initialization(self, mock_openai):
        """Test OpenAI model initialization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        config = OpenAIConfig(
            name="test_openai",
            backend="openai",
            api_key="test_key",
            model_name="gpt-3.5-turbo"
        )
        
        model = OpenAIModel(config)
        assert model.config.api_key == "test_key"
        assert model.config.model_name == "gpt-3.5-turbo"
        mock_openai.assert_called_once()
    
    @patch('macd.models.openai_model.OpenAI')
    def test_openai_model_generate(self, mock_openai):
        """Test OpenAI model generation."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 10
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 5
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        config = OpenAIConfig(
            name="test_openai",
            backend="openai",
            api_key="test_key",
            model_name="gpt-3.5-turbo"
        )
        
        model = OpenAIModel(config)
        strategy = {"generation": {"temperature": 0.7}}
        result = model.generate("Test prompt", strategy)
        
        assert isinstance(result, GenerationResult)
        assert result.text == "Test response"
        assert result.tokens_used == 10
        assert result.finish_reason == "stop"
    
    def test_openai_model_prompt_style(self):
        """Test OpenAI model prompt style formatting."""
        with patch('macd.models.openai_model.OpenAI'):
            config = OpenAIConfig(
                name="test_openai",
                backend="openai",
                api_key="test_key",
                model_name="gpt-3.5-turbo"
            )
            
            model = OpenAIModel(config)
            
            # Test different prompt styles
            styles = ["cot", "concise", "bullet", "json", "detailed"]
            for style in styles:
                strategy = {"prompt_style": style}
                formatted = model._apply_prompt_style("Test question", strategy)
                assert isinstance(formatted, str)
                assert len(formatted) > 0


class TestHuggingFaceModel:
    """Test HuggingFaceModel implementation."""
    
    def test_huggingface_config_validation(self):
        """Test HuggingFace config validation."""
        # Valid config
        config = HuggingFaceConfig(
            name="test_hf",
            backend="huggingface",
            model_path="microsoft/DialoGPT-small"
        )
        assert config.model_path == "microsoft/DialoGPT-small"
        
        # Invalid config (missing model_path)
        with pytest.raises(ValueError):
            HuggingFaceConfig(
                name="test_hf",
                backend="huggingface"
            )
    
    @patch('macd.models.huggingface_model.AutoTokenizer')
    @patch('macd.models.huggingface_model.AutoModelForCausalLM')
    def test_huggingface_model_initialization(self, mock_model, mock_tokenizer):
        """Test HuggingFace model initialization."""
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        config = HuggingFaceConfig(
            name="test_hf",
            backend="huggingface",
            model_path="microsoft/DialoGPT-small"
        )
        
        model = HuggingFaceModel(config)
        assert model.config.model_path == "microsoft/DialoGPT-small"
        mock_tokenizer.assert_called_once()
        mock_model.assert_called_once()
    
    def test_huggingface_model_prompt_style(self):
        """Test HuggingFace model prompt style formatting."""
        with patch('macd.models.huggingface_model.AutoTokenizer'), \
             patch('macd.models.huggingface_model.AutoModelForCausalLM'):
            
            config = HuggingFaceConfig(
                name="test_hf",
                backend="huggingface",
                model_path="microsoft/DialoGPT-small"
            )
            
            model = HuggingFaceModel(config)
            
            # Test different prompt styles
            styles = ["cot", "concise", "bullet", "json", "detailed"]
            for style in styles:
                strategy = {"prompt_style": style}
                formatted = model._apply_prompt_style("Test question", strategy)
                assert isinstance(formatted, str)
                assert len(formatted) > 0


class TestModelFactory:
    """Test ModelFactory implementation."""
    
    def test_model_factory_create_mock(self):
        """Test creating mock model via factory."""
        config_dict = {
            "name": "test_mock",
            "backend": "mock",
            "seed": 42
        }
        
        model = ModelFactory.create_model(config_dict)
        assert isinstance(model, MockModel)
        assert model.config.name == "test_mock"
        assert model.config.backend == "mock"
    
    @patch('macd.models.openai_model.OpenAI')
    def test_model_factory_create_openai(self, mock_openai):
        """Test creating OpenAI model via factory."""
        mock_openai.return_value = Mock()
        
        config_dict = {
            "name": "test_openai",
            "backend": "openai",
            "api_key": "test_key",
            "model_name": "gpt-3.5-turbo"
        }
        
        model = ModelFactory.create_model(config_dict)
        assert isinstance(model, OpenAIModel)
        assert model.config.name == "test_openai"
        assert model.config.backend == "openai"
    
    @patch('macd.models.huggingface_model.AutoTokenizer')
    @patch('macd.models.huggingface_model.AutoModelForCausalLM')
    def test_model_factory_create_huggingface(self, mock_model, mock_tokenizer):
        """Test creating HuggingFace model via factory."""
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        config_dict = {
            "name": "test_hf",
            "backend": "huggingface",
            "model_path": "microsoft/DialoGPT-small"
        }
        
        model = ModelFactory.create_model(config_dict)
        assert isinstance(model, HuggingFaceModel)
        assert model.config.name == "test_hf"
        assert model.config.backend == "huggingface"
    
    def test_model_factory_unsupported_backend(self):
        """Test factory with unsupported backend."""
        config_dict = {
            "name": "test_unsupported",
            "backend": "unsupported"
        }
        
        with pytest.raises(ValueError, match="Unsupported model backend"):
            ModelFactory.create_model(config_dict)
    
    def test_model_factory_list_backends(self):
        """Test listing available backends."""
        backends = ModelFactory.list_backends()
        assert "mock" in backends
        assert "openai" in backends
        assert "huggingface" in backends
    
    def test_model_factory_register_custom(self):
        """Test registering custom model."""
        class CustomModel(MockModel):
            pass
        
        class CustomConfig(MockConfig):
            pass
        
        ModelFactory.register_model("custom", CustomModel, CustomConfig)
        
        assert "custom" in ModelFactory.list_backends()
        
        config_dict = {
            "name": "test_custom",
            "backend": "custom"
        }
        
        model = ModelFactory.create_model(config_dict)
        assert isinstance(model, CustomModel)
