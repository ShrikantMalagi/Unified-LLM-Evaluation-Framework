# Unified LLM Evaluation Framework

This project is a framework that routes datasets to existing evaluation libraries under a single, lightweight interface. It does **not** reimplement metrics.

## Quick Start

```powershell
python -m pip install -e .[dev]
pytest -q
python -m examples.qa_eval
```

Install backend extras as needed:

```powershell
python -m pip install -e .[qa]
python -m pip install -e .[rag]
python -m pip install -e .[mlflow]
python -m examples.chat_eval
```

## Unified API

```python
from runners.evaluate import evaluate
from core.model import CallableModel

dataset = [
    {"question": "What is 2+2?", "answer": "4"}
]

model = CallableModel(name="demo-model", fn=lambda prompt: "4")
result = evaluate(dataset, model, task_type="qa")
print(result)
```

Output format (strict):

```json
{
  "task": "qa",
  "model": "demo-model",
  "scores": {
    "relevance": 0.85
  },
  "metadata": {
    "latency": 1.23
  }
}
```

## Experiment Comparison

```python
from core.model import CallableModel
from runners.evaluate import evaluate
from runners.experiments import evaluate_many, compare_results

dataset = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "Capital of France?", "answer": "Paris"},
]

models = [
    CallableModel(name="model-a", fn=lambda prompt: "4" if "2+2" in prompt else "Paris"),
    CallableModel(name="model-b", fn=lambda prompt: "4"),
]

results = evaluate_many(dataset, models, task_type="qa")
comparison = compare_results(results, baseline_model="model-a")
print(comparison)
```

## MLflow Logging (Optional)

```python
from runners.mlflow_logger import log_evaluation, log_comparison

log_evaluation(result)
log_comparison(comparison)
```

## Supported Backends

- DeepEval (QA/chat metrics)
- RAGAS (RAG metrics)
- Code evaluator (executes generated code against tests)
- Optional: MLflow logging (see MLflow Logging)

## Repository Structure

```
Unified-LLM-Evaluation-Framework/
  core/
  adapters/
  metrics/
  runners/
    evaluate.py
    experiments.py
    mlflow_logger.py
  examples/
  tests/
  requirements.txt
  README.md
```

## Dataset Normalization

- QA: `{ "question": "...", "answer": "..." }`
- RAG: `{ "query": "...", "context": "...", "answer": "..." }`
- Code: `{ "prompt": "...", "tests": ["assert ..."] }`
- Chat: `{ "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "answer": "..." }`

All datasets are normalized to `EvalSample` internally.

Chat messages are rendered into a single prompt string before evaluation:

```text
system: Answer in one short sentence.
user: What is the capital of France?
```

Each `messages` item may be either a string or a dict with `role` and `content`. Dict-based messages are the recommended format because they preserve speaker intent explicitly.

## Notes

- Install the repo with `pip install -e .` so imports and tests work without setting `PYTHONPATH` manually.
- Run examples from the repo root with `python -m examples.qa_eval` or `python -m examples.rag_eval`.
- DeepEval metrics vary by version. If a default metric is missing, pass explicit metric objects to `DeepEvalEvaluator`.
- Dependency upgrades for `deepeval` and `ragas` are guarded by adapter compatibility tests; run `pytest -q` after changing versions to catch API drift.
- RAG evaluation requires `context` per sample.
- Code evaluation defaults to a separate Python process with a timeout. For stronger isolation, use `CodeEvaluator(execution_mode="docker")` to run code in a locked-down container.
- Docker mode expects a local Docker daemon and uses `--network none`, `--read-only`, capability drops, PID limits, CPU/memory limits, and tmpfs-backed writable scratch paths.
- Even in Docker mode, generated code should still be treated as untrusted and handled cautiously.
