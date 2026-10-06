import json

import httpx
import pytest

from macd.backends import GenerationRequest, TrainingNotSupported, build_backend
from macd.backends.openai_compat import OpenAICompatBackend


def _req(prompt, **kw):
    return GenerationRequest(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": prompt}], **kw)


def test_mock_is_deterministic():
    a, b = build_backend("mock"), build_backend("mock")
    prompts = [_req(f"Is thing {i} real?") for i in range(10)]
    assert a.generate(prompts) == b.generate(prompts)


def test_mock_learns_from_training_and_can_roll_back(tmp_path):
    mock = build_backend("mock")
    question = "Is the zorblax purple?"
    before = mock.generate([_req(question)])[0]
    target = "no" if before == "yes" else "yes"
    mock.train([{"prompt": question, "target": target, "messages": []}], hyper={}, out_dir=tmp_path, name="a1")
    assert mock.generate([_req(question)])[0] == before  # training alone does not activate
    mock.activate("a1")
    assert mock.generate([_req(question)])[0] == target
    assert mock.generate([_req("Is a zorblax purple?")])[0] == target  # generalises to a near-duplicate
    mock.activate(None)
    assert mock.generate([_req(question)])[0] == before
    assert json.loads((tmp_path / "mock_adapter.json").read_text()) == {question: target}


def test_mock_uses_in_context_exemplars():
    mock = build_backend("mock")
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "Is the zorblax purple today?"},
        {"role": "assistant", "content": '{"answer": "certainly"}'},
        {"role": "user", "content": "Is the zorblax purple?"},
    ]
    out = mock.generate([GenerationRequest(messages=messages, hints={"style": "json"})])[0]
    assert json.loads(out) == {"answer": "certainly"}


def test_mock_token_budget_truncates_reasoning():
    mock = build_backend("mock")
    short = mock.generate([_req("Is x y?", max_new_tokens=16, hints={"style": "cot"})])[0]
    long = mock.generate([_req("Is x y?", max_new_tokens=64, hints={"style": "cot"})])[0]
    assert "Final answer" not in short
    assert "Final answer" in long


def _endpoint(handler, **params):
    return OpenAICompatBackend(
        {
            "base_url": "http://test/v1",
            "model": "m",
            "retry_backoff": 0,
            "transport": httpx.MockTransport(handler),
            **params,
        }
    )


def test_openai_compat_sends_expected_request_and_keeps_order():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": body["messages"][-1]["content"].upper()}}]}
        )

    backend = _endpoint(handler, concurrency=4)
    out = backend.generate([_req(f"q{i}", temperature=0.3, top_p=0.9, max_new_tokens=12) for i in range(8)])
    assert out == [f"Q{i}" for i in range(8)]
    path, body = seen[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "m" and body["max_tokens"] == 12 and body["temperature"] == 0.3 and "seed" in body


def test_openai_compat_retries_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    assert _endpoint(handler).generate([_req("q")]) == ["ok"]
    assert calls["n"] == 3


def test_openai_compat_gives_up_with_clear_error():
    backend = _endpoint(lambda request: httpx.Response(500), max_retries=1)
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        backend.generate([_req("q")])


def test_openai_compat_without_trainer_cannot_train(tmp_path):
    backend = _endpoint(lambda request: httpx.Response(200, json={}))
    assert backend.supports_training is False
    with pytest.raises(TrainingNotSupported):
        backend.train([], hyper={}, out_dir=tmp_path, name="a")


def test_unknown_backend():
    with pytest.raises(ValueError):
        build_backend("telepathy")
