from dataclasses import dataclass, field
import numpy as np
from abc import ABC, abstractmethod
from src.data.tokenizer import WikiTextTokenizer


class Dataset(ABC):
    pass


class IterDataset(Dataset):
    @abstractmethod
    def __iter__(self):
        pass


@dataclass(slots=True)
class WikiTextDataset(IterDataset):
    data_path: str
    window_size: int = 2
    use_dynamic_window: bool = False
    min_count: int = 1
    subsampling_threshold: float = 1e-5
    tokenizer: WikiTextTokenizer = None

    corpus: np.ndarray = field(init=False)  # flat list of words in the data
    offsets: np.ndarray = field(init=False)  # index of i-th line start
    word_counts: np.ndarray = field(init=False)
    sampling_table: np.ndarray = field(init=False)

    def __post_init__(self):
        if not self.tokenizer:
            self.tokenizer = WikiTextTokenizer.from_file(self.data_path, self.min_count)
        self._load_and_tokenize()
        self._build_sampling_table()

    def __iter__(self):
        num_lines = len(self.offsets) - 1
        shuffled_indicies = np.random.permutation(num_lines)

        for idx in shuffled_indicies:
            start_idx = self.offsets[idx]
            end_idx = self.offsets[idx + 1]  # safe(len(offsets) = len(corpus)+1)

            line_arr = self.corpus[start_idx:end_idx]

            if len(line_arr) < 2:
                continue

            probs = self.sampling_table[line_arr]
            rand_vals = np.random.rand(len(line_arr))
            keep_mask = rand_vals < probs

            subsampled_line = line_arr[keep_mask]

            for idx, target in enumerate(subsampled_line):
                if self.use_dynamic_window:
                    window_size = np.random.randint(1, self.window_size + 1)
                else:
                    window_size = self.window_size

                start_wdx = max(0, idx - window_size)
                end_wdx = min(len(subsampled_line), idx + window_size + 1)

                left = subsampled_line[start_wdx:idx]
                right = subsampled_line[idx + 1 : end_wdx]

                context = np.concatenate([left, right])

                if len(context) > 0:
                    yield context, target

    def _build_sampling_table(self):
        self.word_counts = np.bincount(
            self.corpus, minlength=self.tokenizer.vocab_size
        ).astype(np.float32)

        total_words = sum(self.word_counts)

        normalized_freq_arr = (
            self.word_counts / total_words + 1e-9
        )  # epsilon to avoid div0

        t = self.subsampling_threshold
        keep_probs = np.sqrt(t / normalized_freq_arr) + (t / normalized_freq_arr)
        keep_probs = np.clip(keep_probs, a_min=None, a_max=1.0)
        self.sampling_table = keep_probs

    def _load_and_tokenize(self):
        all_words = []
        offsets = [0]
        with open(self.data_path, "r") as f:
            for line in f.readlines():
                ids = self.tokenizer.encode_line(line)
                if not ids:
                    continue

                all_words.extend(ids)
                offsets.append(len(all_words))

        self.corpus = np.array(all_words)
        self.offsets = np.array(offsets)
