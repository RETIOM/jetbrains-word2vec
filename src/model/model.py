import numpy as np
from abc import abstractmethod, ABC


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


class EmbeddingModel(Model):
    @abstractmethod
    def embed(self, *args, **kwargs):
        pass


class CBOW(EmbeddingModel):
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

    def __init__(self, vocab_size: int, n_dim: int = 300):
        self.vocab_size = vocab_size
        self.n_dim = n_dim

        scale = 1.0 / np.sqrt(n_dim)
        self.embeddings = np.random.randn(vocab_size, n_dim) * scale

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
