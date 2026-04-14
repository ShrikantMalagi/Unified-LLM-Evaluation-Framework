from __future__ import annotations

from typing import Any


class OutputValidationError(ValueError):
    pass


_SCORE_KEY_MAP: dict[str, dict[str, str]] = {
    "rag": {
        "answer_relevancy": "answer_relevance",
    },
}


def normalize_and_validate(
    result: dict[str, Any],
    expected_task: str | None = None,
    expected_model: str | None = None,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise OutputValidationError("Evaluation result must be a dict.")

    for key in ("task", "model", "scores", "metadata"):
        if key not in result:
            raise OutputValidationError(f"Missing required output key: '{key}'.")

    task = _ensure_str(result.get("task"), "task")
    model = _ensure_str(result.get("model"), "model")
    scores = result.get("scores")
    metadata = result.get("metadata")

    if not isinstance(scores, dict):
        raise OutputValidationError("'scores' must be a dict.")
    if not isinstance(metadata, dict):
        raise OutputValidationError("'metadata' must be a dict.")

    if expected_task is not None and task.lower().strip() != expected_task.lower().strip():
        raise OutputValidationError(
            f"Output task '{task}' does not match expected task '{expected_task}'."
        )
    if expected_model is not None and model != expected_model:
        raise OutputValidationError(
            f"Output model '{model}' does not match expected model '{expected_model}'."
        )

    normalized_scores = _normalize_scores(task, scores)

    return {
        "task": task,
        "model": model,
        "scores": normalized_scores,
        "metadata": metadata,
    }


def _normalize_scores(task: str, scores: dict[str, Any]) -> dict[str, float | None]:
    key_map = _SCORE_KEY_MAP.get(task.lower().strip(), {})
    normalized: dict[str, float | None] = {}

    for key, value in scores.items():
        target_key = key_map.get(key, key)
        if target_key in normalized and target_key != key:
            continue
        normalized[target_key] = _coerce_score(value, target_key)

    return normalized


def _coerce_score(value: Any, key: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise OutputValidationError(f"Score '{key}' must be numeric.") from exc


def _ensure_str(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OutputValidationError(f"'{key}' must be a non-empty string.")
    return value
