from abc import abstractmethod, ABC


class Model(ABC):
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs): ...

    @abstractmethod
    def params(self):
        pass

    @abstractmethod
    def grads(self):
        pass


class Head(Model):
    pass


class TrainableModel(Model):
    @abstractmethod
    def save(self, params: dict, save_dir: str): ...


class Embedder(Model):
    @abstractmethod
    def embed(self, *args, **kwargs):
        pass


class TrainableEmbedder(TrainableModel):
    embedder: Embedder
    head: Head
