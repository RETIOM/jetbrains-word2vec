from dataclasses import dataclass
from dataset import IterDataset
from typing import Callable
import numpy as np


def collate_fn(
    batch: list[tuple[np.ndarray, int]],
) -> tuple[np.ndarray, np.ndarray]:
    context, target = zip(*batch)
    target_batch = np.stack(target)

    context_batch = np.full((len(context), max(context, key=lambda x: len(x))), -1)

    for idx, arr in enumerate(context):
        context_batch[idx, len(arr)] = arr

    return context_batch, target_batch


@dataclass(slots=True, frozen=True)
class IterDataloader:
    dataset: IterDataset
    batch_size: int
    collate_fn: Callable | None = collate_fn

    def __iter__(self):
        batch = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []
        if batch:
            yield self.collate_fn(batch)
