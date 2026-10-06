"""Real LoRA training through the HF backend, on a tiny random Llama built
locally (no downloads). Skipped when torch/transformers/peft are not installed."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from macd.backends import GenerationRequest, build_backend  # noqa: E402
from macd.core.controller import MetaController  # noqa: E402

from .conftest import qa_task  # noqa: E402

CORPUS = [
    "Is the sky blue? yes",
    "Is water dry? no",
    "Is fire hot? yes",
    "Is ice warm? no",
    "system user assistant : Answer the question as briefly as possible. Reply with only the answer and nothing else.",
]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    path = tmp_path_factory.mktemp("tiny-llama")
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(CORPUS, trainers.WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>"]))
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
    )
    fast.save_pretrained(path)
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=fast.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        initializer_range=0.3,  # big enough that a frozen output layer can still give sharp predictions
        bos_token_id=fast.bos_token_id,
        eos_token_id=fast.eos_token_id,
    )
    transformers.LlamaForCausalLM(config).save_pretrained(path)
    return str(path)


@pytest.fixture
def backend(model_dir):
    return build_backend(
        "hf", {"model_id": model_dir, "device_map": "cpu", "dtype": "float32", "local_files_only": True}
    )


def _req(text):
    return GenerationRequest(messages=[{"role": "user", "content": text}], max_new_tokens=2)


QA = [("Is the sky blue?", "yes"), ("Is water dry?", "no"), ("Is fire hot?", "yes"), ("Is ice warm?", "no")]


def test_generate_is_batched_and_deterministic(backend):
    reqs = [_req("Is the sky blue?"), _req("Is water dry?"), _req("Is fire")]
    first = backend.generate(reqs)
    assert len(first) == 3 and all(isinstance(t, str) for t in first)
    assert backend.generate(reqs) == first
    # batching with padding must not change any single result
    assert [backend.generate([r])[0] for r in reqs] == first


def test_lora_training_learns_and_adapters_swap(backend, tmp_path):
    examples = [{"prompt": q, "target": a, "messages": [{"role": "user", "content": q}]} for q, a in QA]
    reqs = [_req(q) for q, _ in QA]
    base_out = backend.generate(reqs)
    info = backend.train(examples, hyper={"rank": 8, "steps": 200, "lr": 1e-2}, out_dir=tmp_path / "a1", name="a1")
    assert info["final_loss"] < info["first_loss"] * 0.5
    assert (tmp_path / "a1" / "adapter_config.json").exists()
    # the saved folder loads on its own with plain PEFT
    from peft import PeftModel

    base = transformers.AutoModelForCausalLM.from_pretrained(backend.params["model_id"])
    assert PeftModel.from_pretrained(base, str(tmp_path / "a1")) is not None
    assert backend.active_adapter is None
    assert backend.generate(reqs) == base_out  # training alone changes nothing
    backend.activate("a1")
    tuned = backend.generate(reqs)
    assert [t.split()[0] if t else "" for t in tuned] == [a for _, a in QA]  # it learned the answers
    backend.activate(None)
    assert backend.generate(reqs) == base_out  # roll back restores the base model

    # a second adapter can be trained and swapped independently
    backend.train(examples[:2], hyper={"rank": 4, "steps": 5, "lr": 1e-3}, out_dir=tmp_path / "a2", name="a2")
    backend.activate("a1")
    assert backend.generate(reqs) == tuned


def test_full_cycle_runs_on_hf_backend(model_dir, tmp_path):
    model_cfg = {
        "model": {
            "name": "tiny",
            "backend": "hf",
            "params": {"model_id": model_dir, "device_map": "cpu", "dtype": "float32", "local_files_only": True},
        }
    }
    task = qa_task(search={"population": 3, "top_k": 1, "generations": 1})
    ctrl = MetaController(task_cfg=task, model_cfg=model_cfg, seed=0, run_dir=tmp_path / "run")
    record = ctrl.run_cycle()
    assert 0.0 <= record["val"] <= 1.0
    assert "reason" in record["distill"]
