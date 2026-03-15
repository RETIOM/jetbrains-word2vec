from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Any


class Optimizer(ABC):
    lr: float = 0.025

    @abstractmethod
    def step(self, params: list[Any], grads: list[Any]) -> None:
        pass


@dataclass(slots=True)
class SGD(Optimizer):
    lr: float = 0.025

    def step(self, params, grads) -> None:
        for param, grad in zip(params, grads):
            if isinstance(grad, tuple):
                # Sparse gradient update
                indices, values = grad
                np.add.at(param, indices, -self.lr * values)
            else:
                # Dense gradient update (in-place)
                grad *= self.lr
                param -= grad
