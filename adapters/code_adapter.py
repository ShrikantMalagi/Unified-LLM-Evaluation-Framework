from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

from core.dataset import Dataset
from core.evaluator import Evaluator
from core.model import Model

_CODE_EXEC_RUNNER = """
import json
import sys


def main() -> int:
    payload = json.loads(sys.stdin.read())
    code = payload["code"]
    tests = payload["tests"]
    exec_globals = {}

    try:
        exec(code, exec_globals)
    except BaseException as exc:
        sys.stdout.write(
            json.dumps(
                {
                    "status": "code_error",
                    "passed": 0,
                    "total": len(tests),
                    "error_type": type(exc).__name__,
                }
            )
        )
        return 0

    passed = 0
    for test in tests:
        try:
            exec(test, exec_globals)
            passed += 1
        except BaseException:
            continue

    sys.stdout.write(json.dumps({"status": "ok", "passed": passed, "total": len(tests)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _run_code_in_subprocess(code: str, tests: list[str], timeout_seconds: float) -> dict[str, Any]:
    return _run_code_with_command(
        [sys.executable, "-c", _CODE_EXEC_RUNNER],
        code=code,
        tests=tests,
        timeout_seconds=timeout_seconds,
    )


def _build_docker_command(
    docker_binary: str,
    docker_image: str,
    memory_limit: str,
    cpu_limit: str,
    pids_limit: int,
) -> list[str]:
    return [
        docker_binary,
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(pids_limit),
        "--memory",
        memory_limit,
        "--cpus",
        cpu_limit,
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--tmpfs",
        "/workspace:rw,noexec,nosuid,size=64m",
        "--workdir",
        "/workspace",
        docker_image,
        "python",
        "-c",
        _CODE_EXEC_RUNNER,
    ]


def _run_code_in_docker(
    code: str,
    tests: list[str],
    timeout_seconds: float,
    docker_binary: str,
    docker_image: str,
    memory_limit: str,
    cpu_limit: str,
    pids_limit: int,
) -> dict[str, Any]:
    return _run_code_with_command(
        _build_docker_command(
            docker_binary=docker_binary,
            docker_image=docker_image,
            memory_limit=memory_limit,
            cpu_limit=cpu_limit,
            pids_limit=pids_limit,
        ),
        code=code,
        tests=tests,
        timeout_seconds=timeout_seconds,
    )


def _run_code_with_command(command: list[str], code: str, tests: list[str], timeout_seconds: float) -> dict[str, Any]:
    payload = json.dumps({"code": code, "tests": tests})

    try:
        completed = subprocess.run(
            command,
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "passed": 0, "total": len(tests)}
    except OSError as exc:
        return {
            "status": "runner_error",
            "passed": 0,
            "total": len(tests),
            "error_type": type(exc).__name__,
        }

    if completed.returncode != 0:
        return {
            "status": "crash",
            "passed": 0,
            "total": len(tests),
            "returncode": completed.returncode,
        }

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "status": "crash",
            "passed": 0,
            "total": len(tests),
            "returncode": completed.returncode,
        }

    return {
        "status": str(result.get("status", "crash")),
        "passed": int(result.get("passed", 0)),
        "total": int(result.get("total", len(tests))),
    }


class CodeEvaluator(Evaluator):
    def __init__(
        self,
        timeout_seconds: float = 2.0,
        execution_mode: str = "subprocess",
        docker_binary: str = "docker",
        docker_image: str = "python:3.11-slim",
        memory_limit: str = "256m",
        cpu_limit: str = "1.0",
        pids_limit: int = 64,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        mode = execution_mode.lower().strip()
        if mode not in {"subprocess", "docker"}:
            raise ValueError("execution_mode must be either 'subprocess' or 'docker'")
        if pids_limit <= 0:
            raise ValueError("pids_limit must be greater than 0")
        self._timeout_seconds = timeout_seconds
        self._execution_mode = mode
        self._docker_binary = docker_binary
        self._docker_image = docker_image
        self._memory_limit = memory_limit
        self._cpu_limit = cpu_limit
        self._pids_limit = pids_limit

    def _run_code(self, code: str, tests: list[str]) -> dict[str, Any]:
        if self._execution_mode == "docker":
            return _run_code_in_docker(
                code=code,
                tests=tests,
                timeout_seconds=self._timeout_seconds,
                docker_binary=self._docker_binary,
                docker_image=self._docker_image,
                memory_limit=self._memory_limit,
                cpu_limit=self._cpu_limit,
                pids_limit=self._pids_limit,
            )
        return _run_code_in_subprocess(code, tests, self._timeout_seconds)

    def evaluate(self, dataset: Dataset, model: Model) -> dict[str, Any]:
        total = 0
        passed = 0
        code_error_samples = 0
        timed_out_samples = 0
        crashed_samples = 0
        runner_error_samples = 0
        start = time.perf_counter()

        for sample in dataset.samples:
            tests = None
            if sample.metadata:
                tests = sample.metadata.get("tests")
            if not tests:
                continue

            code = model.generate(sample.input)
            execution = self._run_code(code, list(tests))

            total += execution["total"]
            passed += execution["passed"]

            if execution["status"] == "code_error":
                code_error_samples += 1
            elif execution["status"] == "timeout":
                timed_out_samples += 1
            elif execution["status"] == "crash":
                crashed_samples += 1
            elif execution["status"] == "runner_error":
                runner_error_samples += 1

        latency = time.perf_counter() - start
        pass_rate = (passed / total) if total else 0.0

        return {
            "task": dataset.task_type,
            "model": model.name,
            "scores": {"pass_rate": pass_rate},
            "metadata": {
                "latency": latency,
                "total_tests": total,
                "passed": passed,
                "code_error_samples": code_error_samples,
                "timed_out_samples": timed_out_samples,
                "crashed_samples": crashed_samples,
                "runner_error_samples": runner_error_samples,
                "timeout_seconds": self._timeout_seconds,
                "execution_mode": self._execution_mode,
                "docker_image": self._docker_image if self._execution_mode == "docker" else None,
            },
        }
