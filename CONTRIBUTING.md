# Contributing

## Setup

Use an editable install for day-to-day development:

```powershell
python -m pip install -r requirements-dev.txt
```

Install all optional backend dependencies when working on adapter behavior:

```powershell
python -m pip install -r requirements.txt
```

## Tests

Run the default test suite before opening a pull request:

```powershell
python -m ruff check .
python -m mypy
pytest -q
python -m build
python -m twine check dist/*
```

Ruff checks lint/import hygiene, mypy checks package code and examples, `python -m build` validates source/wheel packaging, Twine validates package metadata, and the default pytest suite intentionally stubs optional services and skips real Docker execution.

## Docker Smoke Test

Run the real Docker smoke test only when Docker is installed and the daemon is available:

```powershell
$env:RUN_DOCKER_SMOKE='1'
pytest -q -m docker
```

The GitHub Actions workflow also exposes this as a manual `workflow_dispatch` job.

## Optional Backend Smoke Tests

Run the real optional-backend smoke tests only after installing all backend extras:

```powershell
python -m pip install -r requirements.txt
$env:RUN_BACKEND_SMOKE='1'
pytest -q -m backend
```

These tests import and exercise real DeepEval, RAGAS/datasets, and MLflow packages without relying on external LLM API calls.

## Dependency Updates

Direct optional and dev dependencies are pinned in `constraints.txt`. When updating `deepeval`, `ragas`, or other backend packages:

1. Update `constraints.txt`.
2. Run `pytest -q`.
3. Run `python -m build` and `python -m twine check dist/*`.
4. Run the Docker smoke test if code-evaluation behavior changed.
5. Run optional backend smoke tests if adapter behavior changed.
6. Note any compatibility behavior changes in the README or changelog.

## Safety

Generated code evaluation is inherently risky. Keep the subprocess runner as the lightweight default, prefer Docker mode for untrusted code, and avoid weakening Docker isolation flags without adding tests that justify the change.
