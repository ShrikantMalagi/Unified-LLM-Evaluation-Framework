from __future__ import annotations

from typing import Any, Iterable

from core.dataset import Dataset


def log_evaluation(
    result: dict[str, Any],
    dataset: Dataset | Iterable[dict[str, Any]] | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    extra_metrics: dict[str, float] | None = None,
) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("mlflow is required for MLflow logging. Install with `pip install mlflow`.") from exc

    def _log() -> None:
        if tags:
            mlflow.set_tags(tags)

        task = result.get("task")
        model_name = result.get("model")
        params: dict[str, Any] = {}
        if task is not None:
            params["task"] = task
        if model_name is not None:
            params["model"] = model_name
        if dataset is not None:
            params["num_samples"] = _dataset_size(dataset)
        if params:
            mlflow.log_params(params)

        metrics: dict[str, float] = {}
        scores = result.get("scores", {}) or {}
        for key, value in scores.items():
            val = _safe_float(value)
            if val is not None:
                metrics[f"score.{key}"] = val

        metadata = result.get("metadata", {}) or {}
        for key, value in metadata.items():
            val = _safe_float(value)
            if val is not None:
                metrics[f"meta.{key}"] = val

        if extra_metrics:
            for key, value in extra_metrics.items():
                val = _safe_float(value)
                if val is not None:
                    metrics[key] = val

        if metrics:
            mlflow.log_metrics(metrics)

        mlflow.log_dict(result, "evaluation_result.json")

    if mlflow.active_run() is None:
        with mlflow.start_run(run_name=run_name):
            _log()
    else:
        _log()


def log_comparison(
    comparison: dict[str, Any],
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("mlflow is required for MLflow logging. Install with `pip install mlflow`.") from exc

    def _log() -> None:
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_dict(comparison, "comparison.json")

        baseline = comparison.get("baseline")
        if isinstance(baseline, str):
            mlflow.log_param("baseline_model", baseline)

        deltas = comparison.get("deltas", {}) or {}
        for model_name, metrics in deltas.items():
            if not isinstance(metrics, dict):
                continue
            for metric, value in metrics.items():
                val = _safe_float(value)
                if val is None:
                    continue
                mlflow.log_metric(f"delta.{model_name}.{metric}", val)

    if mlflow.active_run() is None:
        with mlflow.start_run(run_name=run_name):
            _log()
    else:
        _log()


def _dataset_size(dataset: Dataset | Iterable[dict[str, Any]]) -> int:
    if isinstance(dataset, Dataset):
        return len(dataset.samples)
    try:
        return len(dataset)  # type: ignore[arg-type]
    except TypeError:
        return len(list(dataset))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
