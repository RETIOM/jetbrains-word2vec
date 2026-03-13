from dataclasses import dataclass, field
from src.model.models import Head
import numpy as np


@dataclass(slots=True)
class LinearProjectionHead(Head):
    in_dim: int
    out_dim: int
    W_out: np.ndarray = field(init=False)
    delta_W_out: np.ndarray = field(init=False)
    cache: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        scale = 1.0 / np.sqrt(self.in_dim)
        self.W_out = np.random.randn(self.out_dim, self.in_dim) * scale
        self.delta_W_out = np.zeros_like(self.W_out)

    def forward(self, h, targets=None):
        self.cache["h"] = h
        return h @ self.W_out.T

    def backward(self, delta_Z):
        h = self.cache["h"]

        self.delta_W_out[:] = delta_Z.T @ h

        delta_H = delta_Z @ self.W_out

        return delta_H

    def predict(self, x):
        return x @ self.W_out.T

    def params(self):
        return [self.W_out]

    def grads(self):
        return [self.delta_W_out]


@dataclass(slots=True)
class NegativeSamplingHead(Head):
    in_dim: int
    vocab_size: int
    word_counts: np.ndarray
    k: int = 10

    W_out: np.ndarray | None = field(init=False, default=None)
    noise_dist: np.ndarray = field(init=False)
    delta_W_out: np.ndarray = field(init=False)
    cache: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        smoothed = np.power(self.word_counts.astype(np.float64), 0.75)
        self.noise_dist = smoothed / smoothed.sum()

    def _ensure_w_out(self, n_dim):
        if self.W_out is None:
            self.W_out = np.random.uniform(
                -0.5 / n_dim, 0.5 / n_dim, (self.vocab_size, n_dim)
            )
            self.delta_W_out = np.zeros_like(self.W_out)

    def forward(self, h, targets=None):
        if targets is None:
            raise ValueError("NegativeSamplingHead requires targets in forward pass")

        self._ensure_w_out(h.shape[1])
        neg_ids = np.random.choice(
            self.vocab_size, size=(len(targets), self.k), p=self.noise_dist
        )
        v_pos = self.W_out[targets]
        v_neg = self.W_out[neg_ids]
        pos_dots = np.sum(h * v_pos, axis=1)
        neg_dots = np.einsum("bd,bkd->bk", h, v_neg)
        self.cache = {
            "h": h,
            "targets": targets,
            "neg_ids": neg_ids,
            "v_pos": v_pos,
            "v_neg": v_neg,
        }

        return {"pos_dots": pos_dots, "neg_dots": neg_dots}

    def backward(self, score_grads):
        g_pos = score_grads["g_pos"]
        g_neg = score_grads["g_neg"]

        h = self.cache["h"]
        targets = self.cache["targets"]
        neg_ids = self.cache["neg_ids"]
        v_pos = self.cache["v_pos"]
        v_neg = self.cache["v_neg"]

        delta_H = g_pos[:, None] * v_pos + np.sum(g_neg[:, :, None] * v_neg, axis=1)

        D = self.W_out.shape[1]
        indices = np.concatenate([targets, neg_ids.ravel()])
        values = np.concatenate(
            [g_pos[:, None] * h, (g_neg[:, :, None] * h[:, None, :]).reshape(-1, D)]
        )

        self.delta_W_sparse = (indices, values)
        return delta_H

    def predict(self, x):
        return x @ self.W_out.T

    def params(self):
        return [self.W_out] if self.W_out is not None else []

    def grads(self):
        return [self.delta_W_sparse] if hasattr(self, "delta_W_sparse") else []
