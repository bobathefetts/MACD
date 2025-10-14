"""Tests for evaluation framework."""

import pytest
import numpy as np
from typing import List

from macd.core.evaluator import (
    QAEvaluator,
    SummarizationEvaluator,
    ClassificationEvaluator,
    MultiTaskEvaluator,
    TextSimilarityEvaluator,
    EvalResult
)


class TestQAEvaluator:
    """Test QAEvaluator implementation."""
    
    def test_qa_evaluator_initialization(self):
        """Test QA evaluator initialization."""
        evaluator = QAEvaluator()
        assert evaluator.em_weight == 0.7
        assert evaluator.f1_weight == 0.3
        
        # Test custom config
        config = {"em_weight": 0.8, "f1_weight": 0.2}
        evaluator = QAEvaluator(config)
        assert evaluator.em_weight == 0.8
        assert evaluator.f1_weight == 0.2
    
    def test_qa_evaluator_exact_match(self):
        """Test exact match evaluation."""
        evaluator = QAEvaluator()
        
        # Perfect matches
        predictions = ["yes", "no", "maybe"]
        references = ["yes", "no", "maybe"]
        result = evaluator.evaluate(predictions, references)
        
        assert isinstance(result, EvalResult)
        assert result.primary_metric > 0.9  # Should be very high
        assert "exact_match" in result.metrics
        assert "f1_score" in result.metrics
        assert result.metrics["exact_match"] > 0.9
    
    def test_qa_evaluator_partial_match(self):
        """Test partial match evaluation."""
        evaluator = QAEvaluator()
        
        # Partial matches
        predictions = ["yes", "no", "maybe"]
        references = ["yes", "no", "no"]
        result = evaluator.evaluate(predictions, references)
        
        assert isinstance(result, EvalResult)
        assert 0.0 <= result.primary_metric <= 1.0
        assert result.metrics["exact_match"] == 2/3  # 2 out of 3 exact matches
        assert result.metrics["f1_score"] > 0.0
    
    def test_qa_evaluator_no_match(self):
        """Test no match evaluation."""
        evaluator = QAEvaluator()
        
        # No matches
        predictions = ["yes", "yes", "yes"]
        references = ["no", "no", "no"]
        result = evaluator.evaluate(predictions, references)
        
        assert isinstance(result, EvalResult)
        assert result.primary_metric == 0.0
        assert result.metrics["exact_match"] == 0.0
        assert result.metrics["f1_score"] == 0.0
    
    def test_qa_evaluator_empty_inputs(self):
        """Test evaluator with empty inputs."""
        evaluator = QAEvaluator()
        
        result = evaluator.evaluate([], [])
        assert result.primary_metric == 0.0
        assert result.metrics == {}
    
    def test_qa_evaluator_mismatched_lengths(self):
        """Test evaluator with mismatched input lengths."""
        evaluator = QAEvaluator()
        
        with pytest.raises(ValueError, match="must have the same length"):
            evaluator.evaluate(["yes", "no"], ["yes"])
    
    def test_qa_evaluator_confidence_intervals(self):
        """Test confidence interval calculation."""
        evaluator = QAEvaluator()
        
        predictions = ["yes", "no", "maybe", "yes", "no"]
        references = ["yes", "no", "maybe", "yes", "no"]
        result = evaluator.evaluate(predictions, references)
        
        assert result.confidence_interval is not None
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] <= result.confidence_interval[1]
    
    def test_qa_evaluator_details(self):
        """Test evaluator result details."""
        evaluator = QAEvaluator()
        
        predictions = ["yes", "no"]
        references = ["yes", "no"]
        result = evaluator.evaluate(predictions, references)
        
        assert "em_scores" in result.details
        assert "f1_scores" in result.details
        assert "num_examples" in result.details
        assert result.details["num_examples"] == 2
        assert len(result.details["em_scores"]) == 2
        assert len(result.details["f1_scores"]) == 2


class TestSummarizationEvaluator:
    """Test SummarizationEvaluator implementation."""
    
    def test_summarization_evaluator_initialization(self):
        """Test summarization evaluator initialization."""
        evaluator = SummarizationEvaluator()
        assert evaluator.rouge_weight == 0.4
        assert evaluator.bleu_weight == 0.3
        assert evaluator.f1_weight == 0.3
        
        # Test custom config
        config = {"rouge_weight": 0.5, "bleu_weight": 0.3, "f1_weight": 0.2}
        evaluator = SummarizationEvaluator(config)
        assert evaluator.rouge_weight == 0.5
        assert evaluator.bleu_weight == 0.3
        assert evaluator.f1_weight == 0.2
    
    def test_summarization_evaluator_perfect_match(self):
        """Test summarization evaluator with perfect matches."""
        evaluator = SummarizationEvaluator()
        
        predictions = ["This is a summary.", "Another summary here."]
        references = ["This is a summary.", "Another summary here."]
        result = evaluator.evaluate(predictions, references)
        
        assert isinstance(result, EvalResult)
        assert result.primary_metric > 0.9  # Should be very high
        assert "rouge_l" in result.metrics
        assert "bleu" in result.metrics
        assert "f1_score" in result.metrics
    
    def test_summarization_evaluator_partial_match(self):
        """Test summarization evaluator with partial matches."""
        evaluator = SummarizationEvaluator()
        
        predictions = ["This is a summary.", "Different content here."]
        references = ["This is a summary.", "Another summary here."]
        result = evaluator.evaluate(predictions, references)
        
        assert isinstance(result, EvalResult)
        assert 0.0 <= result.primary_metric <= 1.0
        assert result.metrics["f1_score"] > 0.0  # First one should match
    
    def test_summarization_evaluator_no_match(self):
        """Test summarization evaluator with no matches."""
        evaluator = SummarizationEvaluator()
        
        predictions = ["Completely different text.", "Another different text."]
        references = ["This is a summary.", "Another summary here."]
        result = evaluator.evaluate(predictions, references)
        
        assert isinstance(result, EvalResult)
        assert result.primary_metric >= 0.0  # Should be low but not negative
        assert result.metrics["f1_score"] >= 0.0
    
    def test_summarization_evaluator_details(self):
        """Test summarization evaluator result details."""
        evaluator = SummarizationEvaluator()
        
        predictions = ["Short summary.", "This is a longer summary with more words."]
        references = ["Short summary.", "This is a longer summary with more words."]
        result = evaluator.evaluate(predictions, references)
        
        assert "num_examples" in result.details
        assert "avg_prediction_length" in result.details
        assert "avg_reference_length" in result.details
        assert result.details["num_examples"] == 2
        assert result.details["avg_prediction_length"] > 0
        assert result.details["avg_reference_length"] > 0


class TestClassificationEvaluator:
    """Test ClassificationEvaluator implementation."""
    
    def test_classification_evaluator_initialization(self):
        """Test classification evaluator initialization."""
        evaluator = ClassificationEvaluator()
        assert evaluator.average == "weighted"
        assert evaluator.label_mapping == {}
        
        # Test custom config
        config = {
            "average": "macro",
            "label_mapping": {"yes": "positive", "no": "negative"}
        }
        evaluator = ClassificationEvaluator(config)
        assert evaluator.average == "macro"
        assert evaluator.label_mapping == {"yes": "positive", "no": "negative"}
    
    def test_classification_evaluator_perfect_accuracy(self):
        """Test classification evaluator with perfect accuracy."""
        evaluator = ClassificationEvaluator()
        
        predictions = ["yes", "no", "maybe", "yes"]
        references = ["yes", "no", "maybe", "yes"]
        result = evaluator.evaluate(predictions, references)
        
        assert isinstance(result, EvalResult)
        assert result.primary_metric == 1.0  # Perfect accuracy
        assert result.metrics["accuracy"] == 1.0
        assert result.metrics["precision"] == 1.0
        assert result.metrics["recall"] == 1.0
        assert result.metrics["f1_score"] == 1.0
    
    def test_classification_evaluator_partial_accuracy(self):
        """Test classification evaluator with partial accuracy."""
        evaluator = ClassificationEvaluator()
        
        predictions = ["yes", "no", "maybe", "yes"]
        references = ["yes", "no", "yes", "no"]
        result = evaluator.evaluate(predictions, references)
        
        assert isinstance(result, EvalResult)
        assert result.primary_metric == 0.5  # 2 out of 4 correct
        assert result.metrics["accuracy"] == 0.5
    
    def test_classification_evaluator_with_label_mapping(self):
        """Test classification evaluator with label mapping."""
        config = {
            "label_mapping": {"yes": "positive", "no": "negative", "maybe": "neutral"}
        }
        evaluator = ClassificationEvaluator(config)
        
        predictions = ["yes", "no", "maybe"]
        references = ["positive", "negative", "neutral"]
        result = evaluator.evaluate(predictions, references)
        
        assert isinstance(result, EvalResult)
        assert result.primary_metric == 1.0  # Perfect after mapping
    
    def test_classification_evaluator_per_class_metrics(self):
        """Test classification evaluator per-class metrics."""
        evaluator = ClassificationEvaluator()
        
        predictions = ["yes", "no", "maybe", "yes", "no"]
        references = ["yes", "no", "yes", "yes", "no"]
        result = evaluator.evaluate(predictions, references)
        
        assert "per_class_metrics" in result.details
        per_class = result.details["per_class_metrics"]
        
        # Should have metrics for each unique label
        unique_labels = set(predictions + references)
        assert len(per_class) == len(unique_labels)
        
        for label in unique_labels:
            assert label in per_class
            assert "precision" in per_class[label]
            assert "recall" in per_class[label]
            assert "f1" in per_class[label]
            assert "support" in per_class[label]


class TestMultiTaskEvaluator:
    """Test MultiTaskEvaluator implementation."""
    
    def test_multitask_evaluator_initialization(self):
        """Test multi-task evaluator initialization."""
        qa_evaluator = QAEvaluator()
        summarization_evaluator = SummarizationEvaluator()
        
        task_evaluators = {
            "qa": qa_evaluator,
            "summarization": summarization_evaluator
        }
        
        evaluator = MultiTaskEvaluator(task_evaluators)
        assert len(evaluator.task_evaluators) == 2
        assert "qa" in evaluator.task_evaluators
        assert "summarization" in evaluator.task_evaluators
    
    def test_multitask_evaluator_with_weights(self):
        """Test multi-task evaluator with custom weights."""
        qa_evaluator = QAEvaluator()
        summarization_evaluator = SummarizationEvaluator()
        
        task_evaluators = {
            "qa": qa_evaluator,
            "summarization": summarization_evaluator
        }
        
        task_weights = {"qa": 0.7, "summarization": 0.3}
        
        evaluator = MultiTaskEvaluator(task_evaluators, task_weights)
        assert evaluator.task_weights["qa"] == 0.7
        assert evaluator.task_weights["summarization"] == 0.3
    
    def test_multitask_evaluator_evaluation(self):
        """Test multi-task evaluator evaluation."""
        qa_evaluator = QAEvaluator()
        summarization_evaluator = SummarizationEvaluator()
        
        task_evaluators = {
            "qa": qa_evaluator,
            "summarization": summarization_evaluator
        }
        
        evaluator = MultiTaskEvaluator(task_evaluators)
        
        predictions = ["yes", "This is a summary."]
        references = ["yes", "This is a summary."]
        task_labels = ["qa", "summarization"]
        
        result = evaluator.evaluate(predictions, references, task_labels)
        
        assert isinstance(result, EvalResult)
        assert 0.0 <= result.primary_metric <= 1.0
        assert "qa_exact_match" in result.metrics
        assert "summarization_rouge_l" in result.metrics
    
    def test_multitask_evaluator_mismatched_lengths(self):
        """Test multi-task evaluator with mismatched input lengths."""
        qa_evaluator = QAEvaluator()
        evaluator = MultiTaskEvaluator({"qa": qa_evaluator})
        
        with pytest.raises(ValueError, match="must have the same length"):
            evaluator.evaluate(["yes"], ["yes"], ["qa", "qa"])


class TestTextSimilarityEvaluator:
    """Test TextSimilarityEvaluator base class."""
    
    def test_text_similarity_evaluator_initialization(self):
        """Test text similarity evaluator initialization."""
        evaluator = TextSimilarityEvaluator()
        assert evaluator.config == {}
        
        config = {"remove_punctuation": True}
        evaluator = TextSimilarityEvaluator(config)
        assert evaluator.config["remove_punctuation"] is True
    
    def test_normalize_text(self):
        """Test text normalization."""
        evaluator = TextSimilarityEvaluator()
        
        # Test basic normalization
        text = "  This   is   a   test  "
        normalized = evaluator._normalize_text(text)
        assert normalized == "this is a test"
        
        # Test with punctuation removal
        config = {"remove_punctuation": True}
        evaluator = TextSimilarityEvaluator(config)
        text = "This is a test!"
        normalized = evaluator._normalize_text(text)
        assert normalized == "this is a test"
    
    def test_exact_match(self):
        """Test exact match calculation."""
        evaluator = TextSimilarityEvaluator()
        
        # Perfect match
        assert evaluator._exact_match("yes", "yes") == 1.0
        
        # Case insensitive match
        assert evaluator._exact_match("Yes", "yes") == 1.0
        
        # Whitespace insensitive match
        assert evaluator._exact_match("  yes  ", "yes") == 1.0
        
        # No match
        assert evaluator._exact_match("yes", "no") == 0.0
    
    def test_f1_score(self):
        """Test F1 score calculation."""
        evaluator = TextSimilarityEvaluator()
        
        # Perfect match
        assert evaluator._f1_score("yes", "yes") == 1.0
        
        # Partial match
        f1 = evaluator._f1_score("yes no", "yes maybe")
        assert 0.0 < f1 < 1.0
        
        # No match
        assert evaluator._f1_score("yes", "no") == 0.0
        
        # Empty strings
        assert evaluator._f1_score("", "") == 1.0
        assert evaluator._f1_score("yes", "") == 0.0
        assert evaluator._f1_score("", "yes") == 0.0
    
    def test_ngrams(self):
        """Test n-gram generation."""
        evaluator = TextSimilarityEvaluator()
        
        tokens = ["this", "is", "a", "test"]
        
        # Unigrams
        unigrams = evaluator._get_ngrams(tokens, 1)
        assert unigrams == ["this", "is", "a", "test"]
        
        # Bigrams
        bigrams = evaluator._get_ngrams(tokens, 2)
        assert bigrams == ["this is", "is a", "a test"]
        
        # Trigrams
        trigrams = evaluator._get_ngrams(tokens, 3)
        assert trigrams == ["this is a", "is a test"]
    
    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        evaluator = TextSimilarityEvaluator()
        
        # Test with single value
        scores = [0.8]
        ci = evaluator._calculate_confidence_interval(scores)
        assert ci == (0.8, 0.8)
        
        # Test with multiple values
        scores = [0.7, 0.8, 0.9, 0.8, 0.7]
        ci = evaluator._calculate_confidence_interval(scores)
        assert ci[0] <= ci[1]  # Lower bound <= upper bound
        assert 0.0 <= ci[0] <= 1.0
        assert 0.0 <= ci[1] <= 1.0
        
        # Test with empty list
        scores = []
        ci = evaluator._calculate_confidence_interval(scores)
        assert ci == (0.0, 0.0)