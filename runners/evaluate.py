from __future__ import annotations

from typing import Any, Iterable

from core.dataset import Dataset, normalize_dataset
from core.model import Model
from core.router import get_evaluator


def evaluate(dataset: Dataset | Iterable[dict[str, Any]], model: Model, task_type: str | None = None) -> dict[str, Any]:
    normalized = normalize_dataset(dataset, task_type=task_type)
    evaluator = get_evaluator(normalized.task_type)
    return evaluator.evaluate(normalized, model)
