import pytest

from core.output import OutputValidationError, normalize_and_validate


def test_normalize_and_validate_maps_rag_keys():
    result = {
        "task": "rag",
        "model": "demo",
        "scores": {"answer_relevancy": 0.8, "faithfulness": 0.9},
        "metadata": {"latency": 0.1},
    }

    normalized = normalize_and_validate(result, expected_task="rag", expected_model="demo")

    assert "answer_relevance" in normalized["scores"]
    assert "answer_relevancy" not in normalized["scores"]


def test_normalize_and_validate_preserves_existing_standard_key():
    result = {
        "task": "rag",
        "model": "demo",
        "scores": {"answer_relevance": 0.75, "answer_relevancy": 0.9},
        "metadata": {"latency": 0.1},
    }

    normalized = normalize_and_validate(result)

    assert normalized["scores"]["answer_relevance"] == pytest.approx(0.75)


def test_normalize_and_validate_requires_keys():
    bad = {"task": "qa", "model": "x", "scores": {}}
    with pytest.raises(OutputValidationError):
        normalize_and_validate(bad)
