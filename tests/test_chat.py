import pytest

import core.router as router
from core.dataset import normalize_dataset
from core.model import CallableModel
from runners.evaluate import evaluate


def test_chat_normalization_renders_role_prefixed_prompt():
    dataset = normalize_dataset(
        [
            {
                "messages": [
                    {"role": "system", "content": "Answer in one short sentence."},
                    {"role": "user", "content": "What is the capital of France?"},
                ],
                "answer": "Paris.",
            }
        ],
        task_type="chat",
    )

    sample = dataset.samples[0]
    assert sample.input == "system: Answer in one short sentence.\nuser: What is the capital of France?"
    assert sample.expected_output == "Paris."


def test_chat_normalization_requires_role_and_content():
    with pytest.raises(ValueError, match="messages\\[0\\]\\.role"):
        normalize_dataset(
            [{"messages": [{"content": "Hello"}], "answer": "Hello"}],
            task_type="chat",
        )


def test_chat_normalization_supports_string_messages():
    dataset = normalize_dataset(
        [
            {
                "messages": ["system: Be concise.", "user: What is 2+2?"],
                "answer": "4",
            }
        ],
        task_type="chat",
    )

    assert dataset.samples[0].input == "system: Be concise.\nuser: What is 2+2?"


def test_chat_evaluate_routes_through_chat_registry(monkeypatch):
    class DummyEvaluator:
        def evaluate(self, dataset, model):
            assert dataset.task_type == "chat"
            assert dataset.samples[0].input == "system: Be concise.\nuser: What is 2+2?"
            assert dataset.samples[0].expected_output == "4"
            assert model.generate(dataset.samples[0].input) == "4"
            return {
                "task": "chat",
                "model": model.name,
                "scores": {"relevance": 1.0},
                "metadata": {"latency": 0.0},
            }

    monkeypatch.setitem(router._EVALUATOR_REGISTRY, "chat", DummyEvaluator)

    dataset = [
        {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "answer": "4",
        }
    ]
    model = CallableModel(name="chat-model", fn=lambda _prompt: "4")

    result = evaluate(dataset, model, task_type="chat")

    assert result["task"] == "chat"
    assert result["model"] == "chat-model"
    assert result["scores"]["relevance"] == pytest.approx(1.0)
