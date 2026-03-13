import numpy as np


def default_collate_fn(
    batch: list[tuple[np.ndarray, int]], pad_value: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    sequences, targets = zip(*batch)
    batch_size = len(sequences)
    max_len = max(len(seq) for seq in sequences)

    batch_sequences = np.full(
        (batch_size, max_len), pad_value, dtype=sequences[0].dtype
    )
    batch_targets = np.array(targets, dtype=np.int64)

    for i, seq in enumerate(sequences):
        l = len(seq)  # noqa
        batch_sequences[i, :l] = seq

    return batch_sequences, batch_targets


def wikitext_collate_fn(
    batch: list[tuple[np.ndarray, int]], pad_value: int = -1, window_size=2
) -> tuple[np.ndarray, np.ndarray]:
    context, target = zip(*batch)
    target_batch = np.stack(target)

    context_batch = np.full((len(context), window_size * 2), pad_value)

    for idx, arr in enumerate(context):
        context_batch[idx, : len(arr)] = arr

    return context_batch, target_batch
