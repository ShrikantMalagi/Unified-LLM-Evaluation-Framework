from __future__ import annotations

import time
from typing import Any, Iterable

from core.dataset import Dataset
from core.evaluator import Evaluator
from core.model import Model


class DeepEvalEvaluator(Evaluator):
    def __init__(self, metrics: Iterable[object] | None = None, metric_kwargs: dict[str, Any] | None = None):
        self._metric_specs = list(metrics) if metrics is not None else ["relevance", "correctness", "hallucination"]
        self._metric_kwargs = metric_kwargs or {}

    def _build_metrics(self) -> list[tuple[str, object]]:
        try:
            from deepeval import metrics as deepeval_metrics
        except ImportError as exc:
            raise ImportError(
                "deepeval is required for DeepEvalEvaluator. Install with `pip install deepeval`."
            ) from exc

        resolved: list[tuple[str, object]] = []
        for spec in self._metric_specs:
            if not isinstance(spec, str):
                name = getattr(spec, "name", spec.__class__.__name__).lower()
                resolved.append((name, spec))
                continue

            key = spec.lower().strip()
            if key == "relevance":
                cls = getattr(deepeval_metrics, "AnswerRelevancyMetric", None)
            elif key == "correctness":
                cls = getattr(deepeval_metrics, "CorrectnessMetric", None)
            elif key == "hallucination":
                cls = getattr(deepeval_metrics, "HallucinationMetric", None)
            else:
                cls = None

            if cls is None:
                raise ValueError(
                    f"DeepEval metric '{spec}' is not available in this deepeval version. "
                    "Pass explicit metric objects to DeepEvalEvaluator(metrics=[...])."
                )

            kwargs = self._metric_kwargs.get(key, {})
            resolved.append((key, cls(**kwargs)))

        return resolved

    def evaluate(self, dataset: Dataset, model: Model) -> dict[str, Any]:
        try:
            from deepeval.test_case import LLMTestCase
        except ImportError as exc:
            raise ImportError(
                "deepeval is required for DeepEvalEvaluator. Install with `pip install deepeval`."
            ) from exc

        metrics = self._build_metrics()
        scores: dict[str, list[float]] = {name: [] for name, _ in metrics}

        start = time.perf_counter()
        for sample in dataset.samples:
            actual = model.generate(sample.input)
            test_case = LLMTestCase(
                input=sample.input,
                actual_output=actual,
                expected_output=sample.expected_output,
                context=sample.context,
            )
            for name, metric in metrics:
                metric.measure(test_case)
                score = getattr(metric, "score", None)
                if score is not None:
                    scores[name].append(float(score))

        latency = time.perf_counter() - start
        averaged = {name: (sum(vals) / len(vals)) if vals else 0.0 for name, vals in scores.items()}

        return {
            "task": dataset.task_type,
            "model": model.name,
            "scores": averaged,
            "metadata": {"latency": latency},
        }
