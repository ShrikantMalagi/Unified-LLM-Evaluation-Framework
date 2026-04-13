from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


class Model(ABC):
    name: str

    @abstractmethod
    def generate(self, input: str) -> str:
        raise NotImplementedError


@dataclass
class CallableModel(Model):
    name: str
    fn: Callable[[str], str]

    def generate(self, input: str) -> str:
        return self.fn(input)
