from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class Loss(ABC):
    @abstractmethod
    def forward(self, logits, targets):
        pass

    @abstractmethod
    def backward(self):
        pass


@dataclass(slots=True)
class CrossEntropyLoss(Loss):
    def __call__(self, logits, targets):
        return self.forward(logits, targets)

    def forward(self, logits, targets):
        """Logits and centering used for numerical stability"""
        self.targets = targets

        self.centered = logits - np.max(logits, axis=1, keepdims=True)

        logsumexp = np.log(np.sum(np.exp(self.centered), axis=1))
        target_logits = self.centered[np.arange(len(targets)), targets]

        loss = np.mean(-target_logits + logsumexp)
        return loss

    def backward(self):
        exp = np.exp(self.centered)

        # Softmax
        grad = exp / np.sum(exp, axis=1, keepdims=True)

        grad[np.arange(len(self.targets)), self.targets] -= 1
        grad /= len(self.targets)

        return grad
