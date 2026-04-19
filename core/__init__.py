"""Core abstractions for the framework."""

from .dataset import Dataset, EvalSample, normalize_dataset
from .evaluator import Evaluator
from .model import CallableModel, Model
from .output import OutputValidationError, normalize_and_validate
from .router import get_evaluator, register_evaluator

__all__ = [
    "EvalSample",
    "Dataset",
    "normalize_dataset",
    "Model",
    "CallableModel",
    "Evaluator",
    "get_evaluator",
    "register_evaluator",
    "normalize_and_validate",
    "OutputValidationError",
]
