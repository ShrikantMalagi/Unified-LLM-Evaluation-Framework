import pytest

from runners.experiments import compare_results


def test_compare_results_single_task():
    results = [
        {"task": "rag", "model": "model-a", "scores": {"faithfulness": 0.7, "relevance": 0.5}},
        {"task": "rag", "model": "model-b", "scores": {"faithfulness": 0.9, "relevance": 0.45}},
    ]

    comparison = compare_results(results, baseline_model="model-a")

    assert comparison["task"] == "rag"
    assert comparison["baseline"] == "model-a"
    assert comparison["scores"]["model-a"]["faithfulness"] == pytest.approx(0.7)
    assert comparison["deltas"]["model-b"]["faithfulness"] == pytest.approx(0.2)
    assert comparison["deltas"]["model-b"]["relevance"] == pytest.approx(-0.05)


def test_compare_results_multi_task():
    results = [
        {"task": "qa", "model": "model-a", "scores": {"relevance": 0.6}},
        {"task": "rag", "model": "model-a", "scores": {"faithfulness": 0.8}},
    ]

    comparison = compare_results(results)

    assert "tasks" in comparison
    assert comparison["metadata"]["num_tasks"] == 2
