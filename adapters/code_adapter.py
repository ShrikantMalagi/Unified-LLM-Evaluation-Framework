from __future__ import annotations

import time
from typing import Any

from core.dataset import Dataset
from core.evaluator import Evaluator
from core.model import Model


class CodeEvaluator(Evaluator):
    def evaluate(self, dataset: Dataset, model: Model) -> dict[str, Any]:
        total = 0
        passed = 0
        start = time.perf_counter()

        for sample in dataset.samples:
            tests = None
            if sample.metadata:
                tests = sample.metadata.get("tests")
            if not tests:
                continue

            code = model.generate(sample.input)
            exec_globals: dict[str, Any] = {}
            try:
                exec(code, exec_globals)
            except Exception:
                total += len(tests)
                continue

            for test in tests:
                total += 1
                try:
                    exec(test, exec_globals)
                    passed += 1
                except Exception:
                    continue

        latency = time.perf_counter() - start
        pass_rate = (passed / total) if total else 0.0

        return {
            "task": dataset.task_type,
            "model": model.name,
            "scores": {"pass_rate": pass_rate},
            "metadata": {"latency": latency, "total_tests": total, "passed": passed},
        }
