from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
from src.utils.math_utils import sigmoid


class Loss(ABC):
    @abstractmethod
    def forward(self, logits, targets):
        pass

    @abstractmethod
    def backward(self):
        pass


@dataclass(slots=True, init=False)
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

        return grad


@dataclass(slots=True)
class NegativeSamplingLoss:
    word_counts: np.ndarray
    k: int = 10
    vocab_size: int = field(init=False)
    W_out: np.ndarray | None = field(init=False, default=None)
    noise_dist: np.ndarray = field(init=False)
    delta_W_out: np.ndarray = field(init=False)
    cache: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.vocab_size = len(self.word_counts)
        smoothed = np.power(self.word_counts.astype(np.float64), 0.75)
        self.noise_dist = smoothed / smoothed.sum()

    def __call__(self, h, targets):
        return self.forward(h, targets)

    def _ensure_w_out(self, n_dim):
        if self.W_out is None:
            scale = 1.0 / np.sqrt(n_dim)
            self.W_out = np.random.randn(self.vocab_size, n_dim) * scale

    def forward(self, h, targets):
        self._ensure_w_out(h.shape[1])
        neg_ids = np.random.choice(
            self.vocab_size, size=(len(targets), self.k), p=self.noise_dist
        )
        v_pos = self.W_out[targets]
        v_neg = self.W_out[neg_ids]
        pos_dots = np.sum(h * v_pos, axis=1)
        neg_dots = np.einsum("bd,bkd->bk", h, v_neg)
        neg_dots = np.sum(h[:, None, :] * v_neg, axis=2)
        sig_pos = sigmoid(pos_dots)
        sig_neg = sigmoid(neg_dots)
        loss = -np.mean(
            np.log(sig_pos + 1e-9) + np.sum(np.log(1 - sig_neg + 1e-9), axis=1)
        )
        self.cache = {
            "h": h,
            "targets": targets,
            "neg_ids": neg_ids,
            "v_pos": v_pos,
            "v_neg": v_neg,
            "sig_pos": sig_pos,
            "sig_neg": sig_neg,
        }
        return loss

    def backward(self):
        h = self.cache["h"]
        targets = self.cache["targets"]
        neg_ids = self.cache["neg_ids"]
        v_pos = self.cache["v_pos"]
        v_neg = self.cache["v_neg"]
        sig_pos = self.cache["sig_pos"]
        sig_neg = self.cache["sig_neg"]
        B = len(targets)
        g_pos = sig_pos - 1
        g_neg = sig_neg
        delta_H = g_pos[:, None] * v_pos + np.sum(g_neg[:, :, None] * v_neg, axis=1)
        # delta_H /= B
        self.delta_W_out = np.zeros_like(self.W_out)
        np.add.at(self.delta_W_out, targets, g_pos[:, None] * h)
        D = self.W_out.shape[1]
        np.add.at(
            self.delta_W_out,
            neg_ids.ravel(),
            (g_neg[:, :, None] * h[:, None, :]).reshape(-1, D),
        )
        # self.delta_W_out /= B
        return delta_H

    def params(self):
        return [self.W_out]

    def grads(self):
        return [self.delta_W_out]
