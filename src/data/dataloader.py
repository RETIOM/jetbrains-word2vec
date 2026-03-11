from dataclasses import dataclass
from dataset import IterDataset, Dataset
from utils.collate import collate_fn
from typing import Callable
import numpy as np

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
    collate_fn: Callable = collate_fn

    def __iter__(self):
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if batch:
            yield self.collate_fn(batch)
