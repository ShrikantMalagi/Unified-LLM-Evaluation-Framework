import pytest

import core.router as router


def test_register_evaluator_normalizes_task_type(monkeypatch):
    class CustomEvaluator:
        pass

    original = dict(router._EVALUATOR_REGISTRY)
    monkeypatch.setattr(router, "_EVALUATOR_REGISTRY", original)

    router.register_evaluator("  CuStOm  ", CustomEvaluator)

    evaluator = router.get_evaluator("custom")

    assert isinstance(evaluator, CustomEvaluator)


def test_get_evaluator_rejects_unknown_task_type():
    with pytest.raises(ValueError, match="No evaluator registered"):
        router.get_evaluator("unknown")
