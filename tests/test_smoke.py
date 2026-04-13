import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.model import CallableModel
from runners.evaluate import evaluate


def test_code_eval_pass_rate():
    dataset = [
        {
            "prompt": "def add(a, b):\n    return a + b",
            "tests": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
        }
    ]

    model = CallableModel(name="code-model", fn=lambda prompt: prompt)
    result = evaluate(dataset, model, task_type="code")

    assert result["scores"]["pass_rate"] == 1.0
