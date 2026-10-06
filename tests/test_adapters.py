import pytest

from macd.adapters import Example, extract_answer, format_answer, get_adapter
from macd.core.retrieval import select_exemplars


@pytest.mark.parametrize("style", ["concise", "cot", "bullet", "json"])
def test_format_then_extract_round_trips(style):
    assert extract_answer(format_answer("yes", style), style) == "yes"


def test_extract_returns_empty_when_format_missing():
    assert extract_answer("yes", "json") == ""
    assert extract_answer("I think yes", "cot") == ""
    assert extract_answer("{broken json", "json") == ""


def test_cot_uses_last_final_answer():
    assert extract_answer("Final answer: no\nActually...\nFinal answer: yes", "cot") == "yes"


def test_build_messages_includes_exemplars_in_style():
    adapter = get_adapter("qa")
    shots = [Example("Is ice cold?", "yes")]
    msgs = adapter.build_messages(Example("Is fire hot?", "yes"), "json", shots)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == '{"answer": "yes"}'
    assert msgs[-1]["content"] == "Is fire hot?"


def test_unknown_task_type():
    with pytest.raises(ValueError):
        get_adapter("translation")


def test_exemplar_selection():
    pool = [Example(f"Is item {i} red?", "no") for i in range(6)] + [Example("Do fish swim in water?", "yes")]
    target = Example("Do fish live in water?", "yes")
    similar = select_exemplars(target, pool, k=2, selector="similar")
    assert similar[-1].prompt == "Do fish swim in water?"
    assert select_exemplars(target, pool, 3, "random", seed=1) == select_exemplars(target, pool, 3, "random", seed=1)
    assert select_exemplars(target, pool, 0) == []
    assert target not in select_exemplars(target, [*pool, target], k=20)
