from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Any


class Optimizer(ABC):
    @abstractmethod
    def step(self, params: list[Any], grads: list[Any]) -> None:
        pass


@dataclass(slots=True)
class SGD(Optimizer):
    lr: float = 0.025

    def step(self, params, grads) -> None:
        for param, grad in zip(params, grads):
            np.subtract(param, self.lr * grad, out=param)
