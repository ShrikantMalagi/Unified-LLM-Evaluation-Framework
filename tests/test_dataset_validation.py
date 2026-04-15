import pytest

from core.dataset import normalize_dataset


def test_normalize_dataset_requires_task_type_for_raw_samples():
    with pytest.raises(ValueError, match="task_type is required"):
        normalize_dataset([{"question": "What is 2+2?", "answer": "4"}])


def test_normalize_dataset_requires_qa_answer():
    with pytest.raises(ValueError, match="Missing required field 'answer'"):
        normalize_dataset([{"question": "What is 2+2?"}], task_type="qa")


def test_normalize_dataset_requires_rag_context():
    with pytest.raises(ValueError, match="Missing required field 'context'"):
        normalize_dataset(
            [{"query": "Who wrote The Hobbit?", "answer": "J.R.R. Tolkien"}],
            task_type="rag",
        )


def test_normalize_dataset_requires_code_prompt():
    with pytest.raises(ValueError, match="Missing required field 'prompt'"):
        normalize_dataset([{"tests": ["assert True"]}], task_type="code")


def test_normalize_dataset_rejects_unsupported_task_type():
    with pytest.raises(ValueError, match="Unsupported task_type"):
        normalize_dataset([{"input": "hello"}], task_type="audio")
