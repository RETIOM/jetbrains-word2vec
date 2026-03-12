from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from src.utils.math_utils import sigmoid


class Loss(ABC):
    @abstractmethod
    def forward(self, logits, targets):
        pass

    @abstractmethod
    def backward(self):
        pass

    def __call__(self, logits, targets):
        return self.forward(logits, targets)


@dataclass(slots=True, init=False)
class CrossEntropyLoss(Loss):
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

        return grad


@dataclass(slots=True)
class NegativeSamplingLoss(Loss):
    def forward(self, head_output: dict, targets=None) -> float:
        pos_dots = head_output["pos_dots"]
        neg_dots = head_output["neg_dots"]

        sig_pos = sigmoid(pos_dots)
        sig_neg = sigmoid(neg_dots)

        self.cache = {"sig_pos": sig_pos, "sig_neg": sig_neg}

        loss = -np.mean(
            np.log(sig_pos + 1e-9) + np.sum(np.log(1 - sig_neg + 1e-9), axis=1)
        )
        return loss

    def backward(self):
        sig_pos = self.cache["sig_pos"]
        sig_neg = self.cache["sig_neg"]

        g_pos = sig_pos - 1

        g_neg = sig_neg

        return {"g_pos": g_pos, "g_neg": g_neg}
