import sys
from types import ModuleType, SimpleNamespace

import pytest

from adapters.ragas_adapter import RagasEvaluator
from core.dataset import Dataset, EvalSample
from core.model import CallableModel


def test_ragas_adapter_maps_answer_relevancy(monkeypatch):
    evaluator = RagasEvaluator(metrics=["dummy"])

    class DummyResult:
        def __init__(self):
            self.scores = {"answer_relevancy": 0.7, "faithfulness": 0.9}

    def fake_evaluate(_dataset, metrics):
        assert metrics == ["dummy"]
        return DummyResult()

    class DummyHFDataset:
        @staticmethod
        def from_list(_rows):
            return "hf_dataset"

    monkeypatch.setitem(sys.modules, "ragas", SimpleNamespace(evaluate=fake_evaluate))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=DummyHFDataset))

    dataset = Dataset(
        samples=[
            EvalSample(
                input="Q",
                expected_output="A",
                context=["ctx"],
            )
        ],
        task_type="rag",
    )
    model = CallableModel(name="demo", fn=lambda _prompt: "A")

    result = evaluator.evaluate(dataset, model)

    assert "answer_relevance" in result["scores"]
    assert "answer_relevancy" not in result["scores"]


def test_ragas_extract_scores_accepts_dict_results():
    evaluator = RagasEvaluator()

    scores = evaluator._extract_scores({"faithfulness": "0.8", "answer_relevancy": 0.6})

    assert scores == {"faithfulness": 0.8, "answer_relevancy": 0.6}


def test_ragas_extract_scores_supports_to_pandas_shape():
    evaluator = RagasEvaluator()

    class DummySeries:
        def __init__(self, value):
            self._value = value

        def mean(self):
            return self._value

    class DummyFrame:
        columns = ["question", "faithfulness", "answer_relevancy"]

        def __getitem__(self, key):
            values = {
                "faithfulness": DummySeries(0.75),
                "answer_relevancy": DummySeries(0.65),
            }
            return values[key]

    class DummyResult:
        def to_pandas(self):
            return DummyFrame()

    scores = evaluator._extract_scores(DummyResult())

    assert scores == {"faithfulness": 0.75, "answer_relevancy": 0.65}


def test_ragas_extract_scores_fails_loudly_for_unknown_result_shapes():
    evaluator = RagasEvaluator()

    with pytest.raises(ValueError, match="Unable to extract scores"):
        evaluator._extract_scores(object())


def test_ragas_default_metrics_import_contract(monkeypatch):
    ragas_metrics_module = ModuleType("ragas.metrics")
    ragas_metrics_module.answer_relevancy = "answer_relevancy_metric"
    ragas_metrics_module.context_recall = "context_recall_metric"
    ragas_metrics_module.faithfulness = "faithfulness_metric"

    ragas_module = ModuleType("ragas")
    ragas_module.metrics = ragas_metrics_module

    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", ragas_metrics_module)

    evaluator = RagasEvaluator()

    assert evaluator._default_metrics() == [
        "faithfulness_metric",
        "context_recall_metric",
        "answer_relevancy_metric",
    ]
