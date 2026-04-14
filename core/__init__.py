"""Core abstractions for the framework."""

from .dataset import EvalSample, Dataset, normalize_dataset
from .model import Model, CallableModel
from .evaluator import Evaluator
from .router import get_evaluator, register_evaluator
from .output import normalize_and_validate, OutputValidationError

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
