import pytest

from macd.core.evaluator import QAEvaluator, SummarizationEvaluator, bootstrap_ci, exact_match, f1_unigram


def test_qa_eval_basic():
    res = QAEvaluator().evaluate(["yes", "no"], ["yes", "no"])
    assert res.metric == pytest.approx(1.0)
    assert res.details["em"] == 1.0
    assert res.per_example == [1.0, 1.0]


def test_normalization_ignores_case_and_punctuation():
    assert exact_match("Yes.", "yes") == 1.0
    assert f1_unigram("The  Answer!", "the answer") == 1.0


def test_partial_and_empty():
    assert 0.0 < f1_unigram("yes it is", "yes") < 1.0
    assert f1_unigram("", "yes") == 0.0
    assert QAEvaluator().evaluate([], []).metric == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        QAEvaluator().evaluate(["yes"], ["yes", "no"])


def test_summarization_metric():
    res = SummarizationEvaluator().evaluate(["the bridge reopened"], ["the bridge reopened today"])
    assert 0.8 < res.metric < 1.0


def test_bootstrap_ci_brackets_mean():
    scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    lo, hi = bootstrap_ci(scores, seed=1)
    assert lo <= sum(scores) / len(scores) <= hi
    assert bootstrap_ci(scores, seed=1) == (lo, hi)
