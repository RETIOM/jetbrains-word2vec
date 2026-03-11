import numpy as np


def collate_fn(
    batch: list[tuple[np.ndarray, int]], pad_value: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    context, target = zip(*batch)
    target_batch = np.stack(target)

    context_batch = np.full(
        (len(context), max(context, key=lambda x: len(x))), pad_value
    )

    for idx, arr in enumerate(context):
        context_batch[idx : len(arr)] = arr

    return context_batch, target_batch
