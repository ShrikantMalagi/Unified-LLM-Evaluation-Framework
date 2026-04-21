# Release Process

## Versioning

This project uses semantic versioning:

- Patch releases fix bugs or documentation without changing public behavior.
- Minor releases add backward-compatible features or new adapters.
- Major releases may change public APIs, output schema, or task normalization behavior.

The canonical version is `project.version` in `pyproject.toml`.

## Pre-Release Checklist

1. Update `CHANGELOG.md` with the release date and final notes.
2. Update `project.version` in `pyproject.toml`.
3. Run the default quality gates:

   ```powershell
   python -m ruff check .
   python -m mypy
   pytest -q
   python -m build
   python -m twine check dist/*
   ```

4. Run optional smoke tests when relevant:

   ```powershell
   $env:RUN_DOCKER_SMOKE='1'
   pytest -q -m docker

   $env:RUN_BACKEND_SMOKE='1'
   pytest -q -m backend
   ```

5. Push the branch and verify GitHub Actions.
6. Tag the release, for example `v0.1.0`.

## Publishing

Publishing is intentionally manual for now. After the checklist passes, publish from a clean checkout with a trusted PyPI token:

```powershell
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Do not publish from a dirty working tree.
