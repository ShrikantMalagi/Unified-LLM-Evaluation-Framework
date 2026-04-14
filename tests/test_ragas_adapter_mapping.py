import sys
from types import SimpleNamespace

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
