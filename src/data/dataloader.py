from dataclasses import dataclass
from src.data.dataset import IterDataset, Dataset
from typing import Callable
from src.utils.collate import default_collate_fn

from abc import ABC, abstractmethod


class Dataloader(ABC):
    dataset: Dataset
    batch_size: int
    collate_fn: Callable

    @abstractmethod
    def __iter__(
        self,
    ):
        pass


@dataclass(slots=True, frozen=True)
class IterDataloader(Dataloader):
    dataset: IterDataset
    batch_size: int
    collate_fn: Callable = default_collate_fn

    def __iter__(self):
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self.collate_fn(
                    batch,
                )
                batch = []
        if batch:
            yield self.collate_fn(batch)
