import os
from pathlib import Path
from dataclasses import dataclass, field
from src.model.models import Embedder, Head, TrainableEmbedder
from src.model.inference import InferenceEmbedder
from src.data.tokenizer import Tokenizer
from src.utils.math_utils import softmax
import numpy as np


@dataclass(slots=True)
class CBOWEmbedder(Embedder):
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

        indices = self.cache["batch"][self.cache["mask"]]
        values = delta_per_word[self.cache["mask"]]

        self.delta_E_sparse = (indices, values)

    def predict(self, context):
        lookup = self.embeddings[context]

        h = np.mean(lookup)

        return h

    def params(self):
        return [self.embeddings]

    def grads(self):
        return [self.delta_E_sparse] if hasattr(self, "delta_E_sparse") else []

    def embed(self, word_id):
        return self.embeddings[word_id]


@dataclass(slots=True)
class CBOW(TrainableEmbedder):
    embedder: CBOWEmbedder
    head: Head

    def forward(self, batch, targets=None):
        h = self.embedder.forward(batch)  # the model doesnt see the target
        output = self.head.forward(h, targets)
        return output

    def backward(self, delta_output):
        delta_H = self.head.backward(delta_output)
        self.embedder.backward(delta_H)

    def predict(self, context):
        h = self.embedder.predict(context)
        return self.head.predict(h)

    def embed(self, word_id):
        return self.embedder.embed(word_id)

    def params(self):
        return self.embedder.params() + self.head.params()

    def grads(self):
        return self.embedder.grads() + self.head.grads()

    def save(self, params: dict, save_dir):
        model_params = params["model_params"]
        tokenizer = params["tokenizer"]
        tokenizer.save(save_dir)

        os.makedirs(save_dir, exist_ok=True)
        save_dir = Path(save_dir)

        embeddings = model_params[0]
        w_out = model_params[1] if len(model_params) > 1 else np.array([])

        np.savez(save_dir / "embedder.npz", embeddings=embeddings)
        np.savez(save_dir / "adapter.npz", linear=w_out)


@dataclass(slots=True)
class CBOWInference(InferenceEmbedder):
    embeddings: np.ndarray
    linear: np.ndarray | None
    tokenizer: Tokenizer

    @classmethod
    def from_file(
        cls, model_path: str, tokenizer_path: str, adapter_path: str | None = None
    ):
        tokenizer = Tokenizer.from_file(tokenizer_path)

        embeddings = np.load(model_path, allow_pickle=True)["embeddings"]

        linear = None
        if adapter_path:
            linear = np.load(adapter_path, allow_pickle=True)["linear"]

        return cls(embeddings=embeddings, linear=linear, tokenizer=tokenizer)

    def predict(
        self, context: str | np.ndarray, top_k: int = 1, decode_output: bool = True
    ):
        if not self.linear:
            raise RuntimeError(
                "Prediction not possible for this model. Set linear layer to perform it."
            )

        if isinstance(context, str):
            context = np.array(self.tokenizer.encode(context))

        if not len(context) > 0:
            return []

        lookup = self.embeddings[context]
        h = np.mean(lookup, axis=0)

        y = h @ self.linear.T

        top_k = min(top_k, len(y))

        probs = softmax(y)

        top_indices = np.argsort(y, axis=-1)[-top_k:][::-1]

        if decode_output:
            return [
                (self.tokenizer.decode(word_idx), probs[word_idx])
                for word_idx in top_indices
            ]

        return [(word_idx, probs[word_idx]) for word_idx in top_indices]

    def embed(self, line: str):
        word_indices = np.array(self.tokenizer.encode(line))

        if len(word_indices) == 0:
            return np.empty((0, self.embeddings.shape[1]))

        if len(word_indices) != len(line.strip().split()):
            unknown_words = [
                w for w in line.strip().split() if self.tokenizer.encode(w) is None
            ]
            raise RuntimeError(f"Cannot encode line. Unknown words: {unknown_words}")

        return self.embeddings[word_indices]
