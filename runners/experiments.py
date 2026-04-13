from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from core.dataset import Dataset, normalize_dataset
from core.model import Model
from runners.evaluate import evaluate


def evaluate_many(
    dataset: Dataset | Iterable[dict[str, Any]],
    models: Iterable[Model],
    task_type: str | None = None,
) -> list[dict[str, Any]]:
    normalized = normalize_dataset(dataset, task_type=task_type)
    results: list[dict[str, Any]] = []
    for model in models:
        results.append(evaluate(normalized, model))
    return results


def compare_results(
    results: Iterable[dict[str, Any]],
    baseline_model: str | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        task = str(result.get("task", "unknown"))
        grouped[task].append(result)

    comparisons: dict[str, Any] = {}
    for task, task_results in grouped.items():
        models = [str(r.get("model", "unknown")) for r in task_results]
        metrics: set[str] = set()
        for r in task_results:
            scores = r.get("scores", {}) or {}
            metrics.update(str(k) for k in scores.keys())

        metrics_list = sorted(metrics)
        scores_by_model: dict[str, dict[str, float | None]] = {}
        for idx, r in enumerate(task_results):
            model_name = models[idx]
            scores = r.get("scores", {}) or {}
            scores_by_model[model_name] = {m: _safe_float(scores.get(m)) for m in metrics_list}

        baseline = baseline_model if baseline_model in scores_by_model else (models[0] if models else None)
        deltas: dict[str, dict[str, float | None]] = {}
        if baseline is not None:
            base_scores = scores_by_model.get(baseline, {})
            for model_name in models:
                if model_name == baseline:
                    continue
                deltas[model_name] = {}
                for metric in metrics_list:
                    base_val = base_scores.get(metric)
                    cur_val = scores_by_model.get(model_name, {}).get(metric)
                    if base_val is None or cur_val is None:
                        deltas[model_name][metric] = None
                    else:
                        deltas[model_name][metric] = cur_val - base_val

        comparisons[task] = {
            "task": task,
            "baseline": baseline,
            "models": models,
            "metrics": metrics_list,
            "scores": scores_by_model,
            "deltas": deltas,
        }

    if len(comparisons) == 1:
        return next(iter(comparisons.values()))

    return {
        "tasks": comparisons,
        "metadata": {"num_tasks": len(comparisons), "num_results": sum(len(v["models"]) for v in comparisons.values())},
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
