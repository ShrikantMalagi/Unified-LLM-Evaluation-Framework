# Unified LLM Evaluation Framework

This project is an orchestrator that routes datasets to existing evaluation libraries under a single, lightweight interface. It does **not** reimplement metrics.

## Quick Start

```powershell
$env:PYTHONPATH = "llm-eval-orchestrator"
python llm-eval-orchestrator\examples\qa_eval.py
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

## Supported Backends

- DeepEval (QA/chat metrics)
- RAGAS (RAG metrics)
- Code evaluator (executes generated code against tests)
- Optional: MLflow (not wired yet)

## Repository Structure

```
llm-eval-orchestrator/
  core/
  adapters/
  metrics/
  runners/
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

- `llm-eval-orchestrator` contains the Python modules; add it to `PYTHONPATH` to run examples.
- DeepEval metrics vary by version. If a default metric is missing, pass explicit metric objects to `DeepEvalEvaluator`.
- RAG evaluation requires `context` per sample.
- Code evaluation executes model-generated code. Only run trusted inputs.
