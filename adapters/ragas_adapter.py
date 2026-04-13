from __future__ import annotations

import time
from typing import Any

from core.dataset import Dataset, normalize_contexts
from core.evaluator import Evaluator
from core.model import Model


class RagasEvaluator(Evaluator):
    def __init__(self, metrics: list[object] | None = None):
        self._metrics = metrics

    def _default_metrics(self) -> list[object]:
        try:
            from ragas.metrics import answer_relevancy, context_recall, faithfulness
        except ImportError as exc:
            raise ImportError(
                "ragas is required for RagasEvaluator. Install with `pip install ragas`."
            ) from exc
        return [faithfulness, context_recall, answer_relevancy]

    def _extract_scores(self, result: Any) -> dict[str, float]:
        if isinstance(result, dict):
            return {k: float(v) for k, v in result.items()}

        scores_attr = getattr(result, "scores", None)
        if isinstance(scores_attr, dict):
            return {k: float(v) for k, v in scores_attr.items()}

        to_pandas = getattr(result, "to_pandas", None)
        if callable(to_pandas):
            df = to_pandas()
            metric_scores: dict[str, float] = {}
            for col in df.columns:
                if col in {"question", "answer", "contexts", "ground_truth"}:
                    continue
                series = df[col]
                try:
                    metric_scores[col] = float(series.mean())
                except Exception:
                    continue
            if metric_scores:
                return metric_scores

        raise ValueError("Unable to extract scores from ragas result")

    def evaluate(self, dataset: Dataset, model: Model) -> dict[str, Any]:
        try:
            from ragas import evaluate as ragas_evaluate
            from datasets import Dataset as HFDataset
        except ImportError as exc:
            raise ImportError(
                "ragas (and datasets) are required for RagasEvaluator. Install with `pip install ragas datasets`."
            ) from exc

        metrics = self._metrics or self._default_metrics()
        normalized = normalize_contexts(dataset)

        rows: list[dict[str, Any]] = []
        start = time.perf_counter()
        for sample in normalized.samples:
            if not sample.context:
                raise ValueError("RAG evaluation requires context for each sample")
            answer = model.generate(sample.input)
            rows.append(
                {
                    "question": sample.input,
                    "answer": answer,
                    "contexts": sample.context,
                    "ground_truth": sample.expected_output or "",
                }
            )

        hf_dataset = HFDataset.from_list(rows)
        result = ragas_evaluate(hf_dataset, metrics=metrics)
        scores = self._extract_scores(result)
        if "answer_relevancy" in scores and "answer_relevance" not in scores:
            scores["answer_relevance"] = scores.pop("answer_relevancy")
        latency = time.perf_counter() - start

        return {
            "task": dataset.task_type,
            "model": model.name,
            "scores": scores,
            "metadata": {"latency": latency},
        }
