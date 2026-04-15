import sys
from types import ModuleType

from core.dataset import Dataset, EvalSample
from runners.mlflow_logger import log_comparison, log_evaluation


def test_log_evaluation_starts_run_and_logs_params_metrics_and_artifact(monkeypatch):
    calls = {
        "start_runs": [],
        "tags": [],
        "params": [],
        "metrics": [],
        "dicts": [],
        "log_param": [],
        "log_metric": [],
    }

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    mlflow_module = ModuleType("mlflow")
    mlflow_module.active_run = lambda: None
    mlflow_module.start_run = lambda run_name=None: calls["start_runs"].append(run_name) or DummyRun()
    mlflow_module.set_tags = lambda tags: calls["tags"].append(tags)
    mlflow_module.log_params = lambda params: calls["params"].append(params)
    mlflow_module.log_metrics = lambda metrics: calls["metrics"].append(metrics)
    mlflow_module.log_dict = lambda payload, path: calls["dicts"].append((payload, path))
    mlflow_module.log_param = lambda key, value: calls["log_param"].append((key, value))
    mlflow_module.log_metric = lambda key, value: calls["log_metric"].append((key, value))

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)

    result = {
        "task": "qa",
        "model": "demo-model",
        "scores": {"relevance": 0.8, "ignored": "not-a-number"},
        "metadata": {"latency": 1.2, "tokens": None},
    }
    dataset = Dataset(samples=[EvalSample(input="Q1"), EvalSample(input="Q2")], task_type="qa")

    log_evaluation(
        result,
        dataset=dataset,
        run_name="qa-run",
        tags={"env": "test"},
        extra_metrics={"custom.metric": 3.5, "custom.bad": "skip"},
    )

    assert calls["start_runs"] == ["qa-run"]
    assert calls["tags"] == [{"env": "test"}]
    assert calls["params"] == [{"task": "qa", "model": "demo-model", "num_samples": 2}]
    assert calls["metrics"] == [{"score.relevance": 0.8, "meta.latency": 1.2, "custom.metric": 3.5}]
    assert calls["dicts"] == [(result, "evaluation_result.json")]


def test_log_comparison_uses_active_run_and_logs_baseline_and_deltas(monkeypatch):
    calls = {
        "start_runs": [],
        "tags": [],
        "dicts": [],
        "log_param": [],
        "log_metric": [],
    }

    mlflow_module = ModuleType("mlflow")
    mlflow_module.active_run = lambda: object()
    mlflow_module.start_run = lambda run_name=None: calls["start_runs"].append(run_name)
    mlflow_module.set_tags = lambda tags: calls["tags"].append(tags)
    mlflow_module.log_dict = lambda payload, path: calls["dicts"].append((payload, path))
    mlflow_module.log_param = lambda key, value: calls["log_param"].append((key, value))
    mlflow_module.log_metric = lambda key, value: calls["log_metric"].append((key, value))

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_module)

    comparison = {
        "task": "qa",
        "baseline": "model-a",
        "deltas": {
            "model-b": {"relevance": 0.1, "correctness": None},
        },
    }

    log_comparison(comparison, run_name="comparison-run", tags={"suite": "ci"})

    assert calls["start_runs"] == []
    assert calls["tags"] == [{"suite": "ci"}]
    assert calls["dicts"] == [(comparison, "comparison.json")]
    assert calls["log_param"] == [("baseline_model", "model-a")]
    assert calls["log_metric"] == [("delta.model-b.relevance", 0.1)]
