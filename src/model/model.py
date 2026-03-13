import numpy as np
from abc import abstractmethod, ABC
from dataclasses import dataclass, field


class Model(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def grads(self):
        pass


class Encoder(Model):
    @abstractmethod
    def embed(self, *args, **kwargs):
        pass


class Head(Model):
    pass


@dataclass(slots=True)
class LinearProjectionHead(Model):
    in_dim: int
    out_dim: int
    W_out: np.ndarray = field(init=False)
    delta_W_out: np.ndarray = field(init=False)
    cache: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        scale = 1.0 / np.sqrt(self.in_dim)
        self.W_out = np.random.randn(self.out_dim, self.in_dim) * scale

    def forward(self, h, targets=None):
        self.cache["h"] = h
        return h @ self.W_out.T

    def backward(self, delta_Z):
        h = self.cache["h"]

        self.delta_W_out = delta_Z.T @ h

        delta_H = delta_Z @ self.W_out

        return delta_H

    def params(self):
        return [self.W_out]

    def grads(self):
        return [self.delta_W_out]


@dataclass(slots=True)
class NegativeSamplingHead(Model):
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

        self.delta_W_out.fill(0) # Reset gradients
        np.add.at(self.delta_W_out, targets, g_pos[:, None] * h)

        D = self.W_out.shape[1]
        np.add.at(
            self.delta_W_out,
            neg_ids.ravel(),
            (g_neg[:, :, None] * h[:, None, :]).reshape(-1, D),
        )
        return delta_H

    def params(self):
        return [self.W_out] if self.W_out is not None else []

    def grads(self):
        return [self.delta_W_out] if hasattr(self, "delta_W_out") else []


@dataclass(slots=True)
class CBOWEncoder(Encoder):
    """
    The model consits of 3 layers:
        input: 2*window_size
        projection: n_dim
        output: vocab_size

    (B - batch size; W - window size; V - vocab size; N - dim)
    That means the flow of the data is as follows:
        indicies -> one-hot avg(x) -> x^T @ W1 (h) -> (u) -> softmax (y)
        (B, W*2) -> <proccess each one by one>                           -> (B, V)
    """

    vocab_size: int
    n_dim: int = 300

    embeddings: np.ndarray = field(init=False)
    cache: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        scale = 1.0 / np.sqrt(self.n_dim)
        self.embeddings = np.random.randn(self.vocab_size, self.n_dim) * scale

    def forward(self, batch: np.ndarray):
        """Batch is an array with -1 for padding"""
        mask = batch != -1
        safe_batch = np.where(batch == -1, 0, batch)

        lookup = self.embeddings[safe_batch]
        lookup *= mask[..., None]

        h = np.sum(lookup, axis=1) / np.sum(mask, axis=1, keepdims=True)

        self.cache = {"batch": batch, "mask": mask, "H": h}

        return h

    def backward(self, delta_H):
        """
        Z - output of last layer (batch, vocab)
        H - output of hidden layer (batch, embed_dim)
        W - linear weight matrix (embed_dim, vocab)
        E - embeddings weight matrix (vocab, embed_dim)
        """
        num_words = self.cache["mask"].sum(axis=1, keepdims=True)[:, :, None]

        delta_per_word = delta_H[:, None, :] / num_words

        delta_per_word = delta_per_word * self.cache["mask"][:, :, None]

        self.delta_E = np.zeros_like(self.embeddings)
        np.add.at(
            self.delta_E,
            self.cache["batch"][self.cache["mask"]],
            delta_per_word[self.cache["mask"]],
        )

    def params(self):
        return [self.embeddings]

    def grads(self):
        return [self.delta_E]

    def embed(self, word_id):
        return self.embeddings[word_id]


class TrainableModel(Model):
    encoder: Encoder
    head: Head


@dataclass(slots=True)
class CBOW(TrainableModel):
    encoder: CBOWEncoder
    head: Head

    def forward(self, batch, targets=None):
        h = self.encoder.forward(batch)  # the model doesnt see the target
        output = self.head.forward(h, targets)
        return output

    def backward(self, delta_output):
        delta_H = self.head.backward(delta_output)
        self.encoder.backward(delta_H)

    def params(self):
        return self.encoder.params() + self.head.params()

    def grads(self):
        return self.encoder.grads() + self.head.grads()
