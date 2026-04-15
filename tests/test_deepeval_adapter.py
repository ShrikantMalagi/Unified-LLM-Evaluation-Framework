import sys
from types import ModuleType

import pytest

from adapters.deepeval_adapter import DeepEvalEvaluator
from core.model import CallableModel
from runners.evaluate import evaluate


def test_qa_evaluate_routes_through_deepeval_and_averages_scores(monkeypatch):
    captured_cases = []
    metric_inits = []

    class FakeMetric:
        def __init__(self, name, scores, **kwargs):
            self.name = name
            self._scores = iter(scores)
            self.score = None
            metric_inits.append((name, kwargs))

        def measure(self, test_case):
            captured_cases.append(test_case)
            self.score = next(self._scores)

    metrics_module = ModuleType("deepeval.metrics")
    metrics_module.AnswerRelevancyMetric = lambda **kwargs: FakeMetric("relevance", [0.8, 0.6], **kwargs)
    metrics_module.CorrectnessMetric = lambda **kwargs: FakeMetric("correctness", [1.0, 0.5], **kwargs)
    metrics_module.HallucinationMetric = lambda **kwargs: FakeMetric("hallucination", [0.2, 0.4], **kwargs)

    deepeval_module = ModuleType("deepeval")
    deepeval_module.metrics = metrics_module

    class DummyLLMTestCase:
        def __init__(self, input, actual_output, expected_output, context):
            self.input = input
            self.actual_output = actual_output
            self.expected_output = expected_output
            self.context = context

    test_case_module = ModuleType("deepeval.test_case")
    test_case_module.LLMTestCase = DummyLLMTestCase

    monkeypatch.setitem(sys.modules, "deepeval", deepeval_module)
    monkeypatch.setitem(sys.modules, "deepeval.test_case", test_case_module)

    dataset = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Capital of France?", "answer": "Paris"},
    ]
    model = CallableModel(name="qa-model", fn=lambda prompt: "4" if "2+2" in prompt else "Paris")

    result = evaluate(dataset, model, task_type="qa")

    assert result["task"] == "qa"
    assert result["model"] == "qa-model"
    assert result["scores"]["relevance"] == pytest.approx(0.7)
    assert result["scores"]["correctness"] == pytest.approx(0.75)
    assert result["scores"]["hallucination"] == pytest.approx(0.3)
    assert metric_inits == [("relevance", {}), ("correctness", {}), ("hallucination", {})]
    assert captured_cases[0].input == "What is 2+2?"
    assert captured_cases[0].actual_output == "4"
    assert captured_cases[0].expected_output == "4"
    assert [case.input for case in captured_cases] == [
        "What is 2+2?",
        "What is 2+2?",
        "What is 2+2?",
        "Capital of France?",
        "Capital of France?",
        "Capital of France?",
    ]


def test_deepeval_evaluator_passes_metric_kwargs(monkeypatch):
    seen_kwargs = {}

    class FakeMetric:
        def __init__(self, **kwargs):
            seen_kwargs.update(kwargs)
            self.score = 1.0

        def measure(self, _test_case):
            return None

    metrics_module = ModuleType("deepeval.metrics")
    metrics_module.AnswerRelevancyMetric = FakeMetric

    deepeval_module = ModuleType("deepeval")
    deepeval_module.metrics = metrics_module

    class DummyLLMTestCase:
        def __init__(self, input, actual_output, expected_output, context):
            self.input = input
            self.actual_output = actual_output
            self.expected_output = expected_output
            self.context = context

    test_case_module = ModuleType("deepeval.test_case")
    test_case_module.LLMTestCase = DummyLLMTestCase

    monkeypatch.setitem(sys.modules, "deepeval", deepeval_module)
    monkeypatch.setitem(sys.modules, "deepeval.test_case", test_case_module)

    evaluator = DeepEvalEvaluator(
        metrics=["relevance"],
        metric_kwargs={"relevance": {"threshold": 0.5, "model": "judge-model"}},
    )
    model = CallableModel(name="qa-model", fn=lambda _prompt: "4")

    result = evaluator.evaluate(
        dataset=type("D", (), {"samples": [type("S", (), {"input": "Q", "expected_output": "4", "context": None})()], "task_type": "qa"})(),
        model=model,
    )

    assert seen_kwargs == {"threshold": 0.5, "model": "judge-model"}
    assert result["scores"]["relevance"] == pytest.approx(1.0)
