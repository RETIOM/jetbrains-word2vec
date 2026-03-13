from abc import ABC, abstractmethod


class InferenceModel(ABC):
    @abstractmethod
    def predict(self, *args, **kwargs):
        pass


class InferenceEmbedder(InferenceModel):
    @abstractmethod
    def embed(self, *args, **kwargs):
        pass

    @classmethod
    def from_file(cls, *args, **kwargs):
        pass
