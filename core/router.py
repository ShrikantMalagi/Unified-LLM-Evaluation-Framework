from __future__ import annotations

from collections.abc import Callable

from adapters.code_adapter import CodeEvaluator
from adapters.deepeval_adapter import DeepEvalEvaluator
from adapters.ragas_adapter import RagasEvaluator

_EVALUATOR_REGISTRY: dict[str, Callable[[], object]] = {
    "rag": RagasEvaluator,
    "qa": DeepEvalEvaluator,
    "chat": DeepEvalEvaluator,
    "code": CodeEvaluator,
}


def register_evaluator(task_type: str, factory: Callable[[], object]) -> None:
    key = task_type.lower().strip()
    _EVALUATOR_REGISTRY[key] = factory


def get_evaluator(task_type: str):
    key = task_type.lower().strip()
    if key not in _EVALUATOR_REGISTRY:
        raise ValueError(f"No evaluator registered for task_type='{task_type}'")
    return _EVALUATOR_REGISTRY[key]()
