
import re
import math
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from loguru import logger


@dataclass
class EvalResult:
    """Result from evaluation with comprehensive metrics."""
    primary_metric: float
    metrics: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    confidence_interval: Optional[Tuple[float, float]] = None
    
    @property
    def metric(self) -> float:
        """Backward compatibility property."""
        return self.primary_metric


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def evaluate(self, predictions: List[str], references: List[str]) -> EvalResult:
        """Evaluate predictions against references."""
        pass
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for evaluation."""
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        # Remove punctuation for some metrics
        if self.config.get("remove_punctuation", False):
            text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _calculate_confidence_interval(self, scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for scores."""
        if len(scores) < 2:
            return (scores[0], scores[0]) if scores else (0.0, 0.0)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1)
        n = len(scores)
        
        # Use t-distribution for small samples
        if n < 30:
            from scipy import stats
            t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin = t_val * (std_score / math.sqrt(n))
        else:
            # Use normal distribution for large samples
            z_val = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
            margin = z_val * (std_score / math.sqrt(n))
        
        return (mean_score - margin, mean_score + margin)


class TextSimilarityEvaluator(BaseEvaluator):
    """Evaluator for text similarity tasks."""
    
    def _exact_match(self, pred: str, ref: str) -> float:
        """Calculate exact match score."""
        return 1.0 if self._normalize_text(pred) == self._normalize_text(ref) else 0.0
    
    def _f1_score(self, pred: str, ref: str, ngram: int = 1) -> float:
        """Calculate F1 score for n-grams."""
        pred_tokens = self._normalize_text(pred).split()
        ref_tokens = self._normalize_text(ref).split()
        
        if ngram > 1:
            pred_tokens = self._get_ngrams(pred_tokens, ngram)
            ref_tokens = self._get_ngrams(ref_tokens, ngram)
        
        if not pred_tokens and not ref_tokens:
            return 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        
        # Calculate precision and recall
        common = sum((pred_counter & ref_counter).values())
        precision = common / len(pred_tokens) if pred_tokens else 0.0
        recall = common / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Get n-grams from tokens."""
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def _bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score approximation."""
        total_score = 0.0
        for pred, ref in zip(predictions, references):
            pred_tokens = self._normalize_text(pred).split()
            ref_tokens = self._normalize_text(ref).split()
            
            if not pred_tokens:
                continue
            
            # Calculate precision for 1-4 grams
            precisions = []
            for n in range(1, 5):
                pred_ngrams = self._get_ngrams(pred_tokens, n)
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                
                if not pred_ngrams:
                    precisions.append(0.0)
                    continue
                
                pred_counter = Counter(pred_ngrams)
                ref_counter = Counter(ref_ngrams)
                
                common = sum((pred_counter & ref_counter).values())
                precision = common / len(pred_ngrams)
                precisions.append(precision)
            
            # Calculate BLEU score
            if all(p > 0 for p in precisions):
                bleu = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
                # Apply brevity penalty
                bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens)))
                total_score += bleu * bp
            else:
                total_score += 0.0
        
        return total_score / len(predictions) if predictions else 0.0
    
    def _rouge_l_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate ROUGE-L score approximation."""
        total_score = 0.0
        for pred, ref in zip(predictions, references):
            pred_tokens = self._normalize_text(pred).split()
            ref_tokens = self._normalize_text(ref).split()
            
            if not pred_tokens or not ref_tokens:
                continue
            
            # Calculate LCS
            lcs_length = self._lcs_length(pred_tokens, ref_tokens)
            
            # Calculate precision and recall
            precision = lcs_length / len(pred_tokens)
            recall = lcs_length / len(ref_tokens)
            
            if precision + recall == 0:
                continue
            
            f1 = 2 * precision * recall / (precision + recall)
            total_score += f1
        
        return total_score / len(predictions) if predictions else 0.0
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]


class QAEvaluator(TextSimilarityEvaluator):
    """Evaluator for Question Answering tasks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Use F1 as primary metric (standard for QA evaluation)
        # EM and F1 should not be weighted summed as they have different scales
        self.use_f1_as_primary = self.config.get("use_f1_as_primary", True)

    def evaluate(self, predictions: List[str], references: List[str]) -> EvalResult:
        """Evaluate QA predictions."""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")

        if not predictions:
            return EvalResult(primary_metric=0.0, metrics={})

        # Calculate individual metrics
        em_scores = [self._exact_match(p, r) for p, r in zip(predictions, references)]
        f1_scores = [self._f1_score(p, r) for p, r in zip(predictions, references)]

        # Calculate aggregate metrics
        em = np.mean(em_scores)
        f1 = np.mean(f1_scores)

        # Calculate confidence intervals
        em_ci = self._calculate_confidence_interval(em_scores)
        f1_ci = self._calculate_confidence_interval(f1_scores)

        # Use F1 as primary metric (standard practice in QA evaluation)
        # F1 already balances precision and recall, no need to combine with EM
        if self.use_f1_as_primary:
            primary_metric = f1
        else:
            # Harmonic mean of EM and F1 (if user explicitly wants combined metric)
            if em + f1 > 0:
                primary_metric = 2 * em * f1 / (em + f1)
            else:
                primary_metric = 0.0

        # Additional metrics
        metrics = {
            "f1_score": f1,
            "exact_match": em,
            "f1_std": np.std(f1_scores),
            "em_std": np.std(em_scores),
        }

        details = {
            "em_scores": em_scores,
            "f1_scores": f1_scores,
            "em_confidence_interval": em_ci,
            "f1_confidence_interval": f1_ci,
            "num_examples": len(predictions),
            "primary_metric_type": "f1" if self.use_f1_as_primary else "harmonic_mean"
        }

        return EvalResult(
            primary_metric=primary_metric,
            metrics=metrics,
            details=details,
            confidence_interval=self._calculate_confidence_interval([primary_metric])
        )


class SummarizationEvaluator(TextSimilarityEvaluator):
    """Evaluator for Summarization tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.rouge_weight = self.config.get("rouge_weight", 0.4)
        self.bleu_weight = self.config.get("bleu_weight", 0.3)
        self.f1_weight = self.config.get("f1_weight", 0.3)
    
    def evaluate(self, predictions: List[str], references: List[str]) -> EvalResult:
        """Evaluate summarization predictions."""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        if not predictions:
            return EvalResult(primary_metric=0.0, metrics={})
        
        # Calculate individual metrics
        f1_scores = [self._f1_score(p, r) for p, r in zip(predictions, references)]
        rouge_l = self._rouge_l_score(predictions, references)
        bleu = self._bleu_score(predictions, references)
        
        # Calculate aggregate metrics
        f1 = np.mean(f1_scores)
        
        # Primary metric (weighted combination)
        primary_metric = (self.rouge_weight * rouge_l + 
                         self.bleu_weight * bleu + 
                         self.f1_weight * f1)
        
        # Additional metrics
        metrics = {
            "rouge_l": rouge_l,
            "bleu": bleu,
            "f1_score": f1,
            "f1_std": np.std(f1_scores),
        }
        
        details = {
            "f1_scores": f1_scores,
            "num_examples": len(predictions),
            "avg_prediction_length": np.mean([len(p.split()) for p in predictions]),
            "avg_reference_length": np.mean([len(r.split()) for r in references]),
        }
        
        return EvalResult(
            primary_metric=primary_metric,
            metrics=metrics,
            details=details
        )


class ClassificationEvaluator(BaseEvaluator):
    """Evaluator for Classification tasks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.label_mapping = self.config.get("label_mapping", {})
        self.average = self.config.get("average", "weighted")
    
    def evaluate(self, predictions: List[str], references: List[str]) -> EvalResult:
        """Evaluate classification predictions."""
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        if not predictions:
            return EvalResult(primary_metric=0.0, metrics={})
        
        # Map labels if mapping is provided
        pred_labels = [self.label_mapping.get(p, p) for p in predictions]
        ref_labels = [self.label_mapping.get(r, r) for r in references]
        
        # Calculate metrics
        accuracy = accuracy_score(ref_labels, pred_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            ref_labels, pred_labels, average=self.average, zero_division=0
        )
        
        # Primary metric (accuracy)
        primary_metric = accuracy
        
        # Additional metrics
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            ref_labels, pred_labels, average=None, zero_division=0
        )
        
        unique_labels = sorted(set(ref_labels + pred_labels))
        per_class_metrics = {}
        for i, label in enumerate(unique_labels):
            if i < len(precision_per_class):
                per_class_metrics[label] = {
                    "precision": precision_per_class[i],
                    "recall": recall_per_class[i],
                    "f1": f1_per_class[i],
                    "support": support_per_class[i],
                }
        
        details = {
            "per_class_metrics": per_class_metrics,
            "num_examples": len(predictions),
            "unique_labels": unique_labels,
        }
        
        return EvalResult(
            primary_metric=primary_metric,
            metrics=metrics,
            details=details
        )


class MultiTaskEvaluator(BaseEvaluator):
    """Evaluator for multi-task scenarios."""
    
    def __init__(self, task_evaluators: Dict[str, BaseEvaluator], 
                 task_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.task_evaluators = task_evaluators
        self.task_weights = task_weights or {task: 1.0 for task in task_evaluators.keys()}
    
    def evaluate(self, predictions: List[str], references: List[str], 
                 task_labels: List[str]) -> EvalResult:
        """Evaluate multi-task predictions."""
        if len(predictions) != len(references) != len(task_labels):
            raise ValueError("All input lists must have the same length")
        
        if not predictions:
            return EvalResult(primary_metric=0.0, metrics={})
        
        # Group by task
        task_groups = {}
        for pred, ref, task in zip(predictions, references, task_labels):
            if task not in task_groups:
                task_groups[task] = {"predictions": [], "references": []}
            task_groups[task]["predictions"].append(pred)
            task_groups[task]["references"].append(ref)
        
        # Evaluate each task
        task_results = {}
        weighted_scores = []
        
        for task, group in task_groups.items():
            if task in self.task_evaluators:
                result = self.task_evaluators[task].evaluate(
                    group["predictions"], group["references"]
                )
                task_results[task] = result
                weight = self.task_weights.get(task, 1.0)
                weighted_scores.append(result.primary_metric * weight)
        
        # Calculate overall metric
        total_weight = sum(self.task_weights.get(task, 1.0) for task in task_groups.keys())
        primary_metric = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
        
        # Aggregate metrics
        all_metrics = {}
        for task, result in task_results.items():
            for metric_name, metric_value in result.metrics.items():
                all_metrics[f"{task}_{metric_name}"] = metric_value
        
        details = {
            "task_results": {task: result.details for task, result in task_results.items()},
            "task_weights": self.task_weights,
            "num_tasks": len(task_groups),
        }
        
        return EvalResult(
            primary_metric=primary_metric,
            metrics=all_metrics,
            details=details
        )


# Backward compatibility
Evaluator = BaseEvaluator
