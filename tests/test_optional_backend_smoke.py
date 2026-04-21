import os
import tempfile
from pathlib import Path

import pytest

from adapters.deepeval_adapter import DeepEvalEvaluator
from adapters.ragas_adapter import RagasEvaluator
from core.dataset import Dataset, EvalSample
from core.model import CallableModel
from runners.mlflow_logger import log_comparison, log_evaluation


def _require_backend_smoke_enabled() -> None:
    if os.environ.get("RUN_BACKEND_SMOKE") != "1":
        pytest.skip("set RUN_BACKEND_SMOKE=1 to run real optional-backend smoke tests")


@pytest.mark.backend
def test_deepeval_backend_import_and_test_case_smoke():
    _require_backend_smoke_enabled()

    pytest.importorskip("deepeval")
    pytest.importorskip("deepeval.test_case")

    class StaticMetric:
        name = "static_score"

        def __init__(self):
            self.score = 0.0

        def measure(self, test_case):
            assert test_case.input == "What is 2+2?"
            assert test_case.actual_output == "4"
            assert test_case.expected_output == "4"
            self.score = 1.0

    evaluator = DeepEvalEvaluator(metrics=[StaticMetric()])
    dataset = Dataset(samples=[EvalSample(input="What is 2+2?", expected_output="4")], task_type="qa")
    model = CallableModel(name="deepeval-smoke-model", fn=lambda _prompt: "4")

    result = evaluator.evaluate(dataset, model)

    assert result["scores"]["static_score"] == pytest.approx(1.0)


@pytest.mark.backend
def test_ragas_backend_default_metrics_and_datasets_smoke():
    _require_backend_smoke_enabled()

    pytest.importorskip("ragas")
    datasets_module = pytest.importorskip("datasets")

    metrics = RagasEvaluator()._default_metrics()
    hf_dataset = datasets_module.Dataset.from_list(
        [
            {
                "question": "Who wrote The Hobbit?",
                "answer": "J.R.R. Tolkien",
                "contexts": ["J.R.R. Tolkien wrote The Hobbit."],
                "ground_truth": "J.R.R. Tolkien",
            }
        ]
    )

    assert len(metrics) == 3
    assert hf_dataset.num_rows == 1


@pytest.mark.backend
def test_mlflow_backend_logging_smoke():
    _require_backend_smoke_enabled()

    mlflow = pytest.importorskip("mlflow")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mlflow.set_tracking_uri(tmp_path.as_uri())

        result = {
            "task": "qa",
            "model": "mlflow-smoke-model",
            "scores": {"relevance": 1.0},
            "metadata": {"latency": 0.01},
        }
        comparison = {
            "task": "qa",
            "baseline": "model-a",
            "deltas": {"model-b": {"relevance": 0.1}},
        }

        log_evaluation(result, dataset=[{"question": "Q", "answer": "A"}], run_name="mlflow-smoke-eval")
        log_comparison(comparison, run_name="mlflow-smoke-comparison")

        assert (tmp_path / "0").exists()
