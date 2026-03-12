from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def step(self, params: list[any], grads: list[any]) -> None:
        pass


# @dataclass(slots=True)
# class SGD(Optimizer):
#     lr: float = 0.001

#     def step(self, params, grads) -> None:
#         for param, grad in zip(params, grads):
#             np.subtract(param, self.lr * grad, out=param)


@dataclass(slots=True)
class SGD(Optimizer):
    lr: float = 0.5
    min_lr: float = 1e-4
    total_steps: int = 100000
    current_step: int = 0

    def step(self, params, grads) -> None:
        effective_lr = max(
            self.min_lr, self.lr * (1 - self.current_step / self.total_steps)
        )
        for param, grad in zip(params, grads):
            np.subtract(param, effective_lr * grad, out=param)
        self.current_step += 1
