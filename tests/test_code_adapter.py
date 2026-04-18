import subprocess

import pytest

import adapters.code_adapter as code_adapter
from adapters.code_adapter import CodeEvaluator
from core.dataset import normalize_dataset
from core.model import CallableModel


def test_code_evaluator_runs_generated_code_in_isolated_process():
    dataset = normalize_dataset(
        [
            {
                "prompt": "write add",
                "tests": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"],
            }
        ],
        task_type="code",
    )
    model = CallableModel(name="code-model", fn=lambda _prompt: "def add(a, b):\n    return a + b")

    result = CodeEvaluator(timeout_seconds=1.0).evaluate(dataset, model)

    assert result["scores"]["pass_rate"] == pytest.approx(1.0)
    assert result["metadata"]["passed"] == 2
    assert result["metadata"]["timed_out_samples"] == 0
    assert result["metadata"]["crashed_samples"] == 0


def test_code_evaluator_reports_code_errors_cleanly():
    dataset = normalize_dataset(
        [
            {
                "prompt": "write add",
                "tests": ["assert add(1, 2) == 3"],
            }
        ],
        task_type="code",
    )
    model = CallableModel(name="code-model", fn=lambda _prompt: "raise RuntimeError('boom')")

    result = CodeEvaluator(timeout_seconds=1.0).evaluate(dataset, model)

    assert result["scores"]["pass_rate"] == pytest.approx(0.0)
    assert result["metadata"]["total_tests"] == 1
    assert result["metadata"]["passed"] == 0
    assert result["metadata"]["code_error_samples"] == 1


def test_code_evaluator_times_out_hanging_code():
    dataset = normalize_dataset(
        [
            {
                "prompt": "hang forever",
                "tests": ["assert True"],
            }
        ],
        task_type="code",
    )
    model = CallableModel(name="code-model", fn=lambda _prompt: "while True:\n    pass")

    result = CodeEvaluator(timeout_seconds=0.1).evaluate(dataset, model)

    assert result["scores"]["pass_rate"] == pytest.approx(0.0)
    assert result["metadata"]["timed_out_samples"] == 1
    assert result["metadata"]["passed"] == 0


def test_code_evaluator_reports_subprocess_crashes_cleanly():
    dataset = normalize_dataset(
        [
            {
                "prompt": "crash hard",
                "tests": ["assert True"],
            }
        ],
        task_type="code",
    )
    model = CallableModel(name="code-model", fn=lambda _prompt: "import os\nos._exit(3)")

    result = CodeEvaluator(timeout_seconds=1.0).evaluate(dataset, model)

    assert result["scores"]["pass_rate"] == pytest.approx(0.0)
    assert result["metadata"]["crashed_samples"] == 1
    assert result["metadata"]["passed"] == 0


def test_code_evaluator_docker_mode_uses_hardened_container_command(monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout='{"status":"ok","passed":1,"total":1}', stderr="")

    monkeypatch.setattr(code_adapter.subprocess, "run", fake_run)

    dataset = normalize_dataset(
        [
            {
                "prompt": "write add",
                "tests": ["assert add(1, 2) == 3"],
            }
        ],
        task_type="code",
    )
    model = CallableModel(name="code-model", fn=lambda _prompt: "def add(a, b):\n    return a + b")

    result = CodeEvaluator(
        timeout_seconds=1.0,
        execution_mode="docker",
        docker_binary="docker",
        docker_image="python:3.11-slim",
    ).evaluate(dataset, model)

    command, kwargs = commands[0]
    assert command[:4] == ["docker", "run", "--rm", "-i"]
    assert "--network" in command and "none" in command
    assert "--read-only" in command
    assert "--cap-drop" in command and "ALL" in command
    assert "--security-opt" in command and "no-new-privileges" in command
    assert "--tmpfs" in command
    assert "python:3.11-slim" in command
    assert kwargs["timeout"] == 1.0
    assert result["scores"]["pass_rate"] == pytest.approx(1.0)
    assert result["metadata"]["execution_mode"] == "docker"
    assert result["metadata"]["docker_image"] == "python:3.11-slim"


def test_code_evaluator_reports_missing_docker_binary_cleanly(monkeypatch):
    def fake_run(_command, **_kwargs):
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr(code_adapter.subprocess, "run", fake_run)

    dataset = normalize_dataset(
        [
            {
                "prompt": "write add",
                "tests": ["assert add(1, 2) == 3"],
            }
        ],
        task_type="code",
    )
    model = CallableModel(name="code-model", fn=lambda _prompt: "def add(a, b):\n    return a + b")

    result = CodeEvaluator(timeout_seconds=1.0, execution_mode="docker").evaluate(dataset, model)

    assert result["scores"]["pass_rate"] == pytest.approx(0.0)
    assert result["metadata"]["runner_error_samples"] == 1
    assert result["metadata"]["passed"] == 0
    assert result["metadata"]["execution_mode"] == "docker"


def test_code_evaluator_rejects_unknown_execution_mode():
    with pytest.raises(ValueError, match="execution_mode"):
        CodeEvaluator(execution_mode="vm")
