from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .dataset import Dataset
from .model import Model


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, dataset: Dataset, model: Model) -> dict[str, Any]:
        raise NotImplementedError
