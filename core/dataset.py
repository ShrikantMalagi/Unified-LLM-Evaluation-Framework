from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class EvalSample:
    input: str
    expected_output: str | None = None
    context: str | list[str] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class Dataset:
    samples: list[EvalSample]
    task_type: str  # "qa", "rag", "code", "chat"
    metadata: dict[str, Any] | None = None


def _ensure_list(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def _require(value: Any, field: str, task_type: str) -> Any:
    if value is None or value == "":
        raise ValueError(f"Missing required field '{field}' for task_type='{task_type}'")
    return value


def _normalize_chat_messages(messages: Any, task_type: str) -> str:
    if not isinstance(messages, list):
        raise ValueError(f"Field 'messages' for task_type='{task_type}' must be a list")

    rendered: list[str] = []
    for idx, message in enumerate(messages):
        if isinstance(message, str):
            rendered.append(message)
            continue

        if not isinstance(message, dict):
            raise ValueError(
                f"messages[{idx}] for task_type='{task_type}' must be a string or dict with role/content"
            )

        role = _require(message.get("role"), f"messages[{idx}].role", task_type)
        content = _require(message.get("content"), f"messages[{idx}].content", task_type)
        rendered.append(f"{role}: {content}")

    return "\n".join(rendered)


def _normalize_sample(raw: dict[str, Any], task_type: str) -> EvalSample:
    task = task_type.lower().strip()

    if task == "qa":
        question = raw.get("question") or raw.get("input") or raw.get("prompt")
        answer = raw.get("answer") or raw.get("expected_output")
        return EvalSample(
            input=_require(question, "question", task_type),
            expected_output=_require(answer, "answer", task_type),
            metadata=raw.get("metadata"),
        )

    if task == "rag":
        query = raw.get("query") or raw.get("question") or raw.get("input")
        context = raw.get("context") or raw.get("contexts")
        answer = raw.get("answer") or raw.get("expected_output") or raw.get("ground_truth")
        return EvalSample(
            input=_require(query, "query", task_type),
            expected_output=_require(answer, "answer", task_type),
            context=_require(context, "context", task_type),
            metadata=raw.get("metadata"),
        )

    if task == "code":
        prompt = raw.get("prompt") or raw.get("input")
        tests = raw.get("tests")
        metadata = raw.get("metadata", {})
        if tests is not None:
            metadata = {**metadata, "tests": tests}
        return EvalSample(
            input=_require(prompt, "prompt", task_type),
            expected_output=None,
            metadata=metadata,
        )

    if task == "chat":
        messages = raw.get("messages")
        if messages is not None:
            rendered = _normalize_chat_messages(messages, task_type)
        else:
            rendered = raw.get("input") or raw.get("prompt") or ""
        expected = raw.get("answer") or raw.get("expected_output")
        return EvalSample(
            input=_require(rendered, "messages", task_type),
            expected_output=expected,
            metadata=raw.get("metadata"),
        )

    raise ValueError(f"Unsupported task_type: {task_type}")


def normalize_dataset(
    dataset: Dataset | Iterable[dict[str, Any]],
    task_type: str | None = None,
) -> Dataset:
    if isinstance(dataset, Dataset):
        return dataset

    if task_type is None:
        raise ValueError("task_type is required when passing raw samples")

    samples = [_normalize_sample(raw, task_type) for raw in dataset]

    return Dataset(samples=samples, task_type=task_type)


def normalize_contexts(dataset: Dataset) -> Dataset:
    normalized = []
    for sample in dataset.samples:
        normalized.append(
            EvalSample(
                input=sample.input,
                expected_output=sample.expected_output,
                context=_ensure_list(sample.context),
                metadata=sample.metadata,
            )
        )
    return Dataset(samples=normalized, task_type=dataset.task_type, metadata=dataset.metadata)
