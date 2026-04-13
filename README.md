# Unified LLM Evaluation Framework

This project is a framework that routes datasets to existing evaluation libraries under a single, lightweight interface. It does **not** reimplement metrics.

## Quick Start

```powershell
$env:PYTHONPATH = (Get-Location)
python .\examples\qa_eval.py
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

All datasets are normalized to `EvalSample` internally.

## Notes

- The repo root contains the Python modules; add it to `PYTHONPATH` to run examples from elsewhere.
- DeepEval metrics vary by version. If a default metric is missing, pass explicit metric objects to `DeepEvalEvaluator`.
- RAG evaluation requires `context` per sample.
- Code evaluation executes model-generated code. Only run trusted inputs.
