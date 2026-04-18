import os
import shutil
import subprocess

import pytest

from adapters.code_adapter import CodeEvaluator
from core.dataset import normalize_dataset
from core.model import CallableModel


def _require_docker_smoke_enabled() -> None:
    if os.environ.get("RUN_DOCKER_SMOKE") != "1":
        pytest.skip("set RUN_DOCKER_SMOKE=1 to run real Docker smoke tests")

    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not installed")

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"Docker daemon is not usable: {type(exc).__name__}")

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        pytest.skip(f"Docker daemon is not usable: {stderr}")


@pytest.mark.docker
def test_code_evaluator_docker_mode_real_smoke():
    _require_docker_smoke_enabled()

    dataset = normalize_dataset(
        [
            {
                "prompt": "write multiply",
                "tests": ["assert multiply(3, 4) == 12"],
            }
        ],
        task_type="code",
    )
    model = CallableModel(name="docker-code-model", fn=lambda _prompt: "def multiply(a, b):\n    return a * b")

    result = CodeEvaluator(
        timeout_seconds=10.0,
        execution_mode="docker",
        docker_image=os.environ.get("CODE_EVAL_DOCKER_IMAGE", "python:3.11-slim"),
    ).evaluate(dataset, model)

    assert result["scores"]["pass_rate"] == pytest.approx(1.0)
    assert result["metadata"]["passed"] == 1
    assert result["metadata"]["execution_mode"] == "docker"
    assert result["metadata"]["runner_error_samples"] == 0
    assert result["metadata"]["timed_out_samples"] == 0
    assert result["metadata"]["crashed_samples"] == 0
