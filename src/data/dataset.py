from dataclasses import dataclass, field
import re
import numpy as np
from collections import Counter
from abc import ABC, abstractmethod


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
    min_freq: int = 1
    subsampling_threshold: float = 1e-5

    corpus: np.ndarray = field(init=False)  # flat list of words in the data
    offsets: np.ndarray = field(init=False)  # index of i-th line start
    word2id: dict[str, int] = field(init=False)
    id2word: dict[int, str] = field(init=False)
    sampling_table: np.ndarray = field(init=False)

    def __post_init__(self):
        self._create_vocab()
        self._load_and_tokenize()

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
                # Dynamic Window Size (as seen in paper)
                dynamic_w = np.random.randint(1, self.window_size + 1)

                start_wdx = max(0, idx - dynamic_w)
                end_wdx = min(len(subsampled_line), idx + dynamic_w + 1)

                left = subsampled_line[start_wdx:idx]
                right = subsampled_line[idx + 1 : end_wdx]

                context = np.concatenate([left, right])

                if len(context) > 0:
                    yield context, target

    def __clean_line(self, line: str) -> str:
        if not (line := line.strip().lower()):
            return ""
        if re.match(r"^=.*=$", line) is not None:
            return ""

        line.replace("<unk>", "")
        return re.sub(r"[^a-z0-9]+", " ", line)

    def __build_sampling_table(self, word_counts) -> np.ndarray:
        vocab_size = len(self.word2id)

        counts_arr = np.zeros(vocab_size, dtype=np.float32)

        for word, count in word_counts.items():
            if word in self.word2id:
                counts_arr[self.word2id[word]] = count

        total_words = np.sum(counts_arr)

        normalized_freq_arr = (
            counts_arr / total_words + 1e-9
        )  # epsilon added to prevent div by 0 err

        keep_probs = np.sqrt(self.subsampling_threshold / normalized_freq_arr) + (
            self.subsampling_threshold / normalized_freq_arr
        )
        keep_probs = np.clip(keep_probs, a_min=None, a_max=1.0)

        return keep_probs

    def _create_vocab(self):
        counts = Counter()

        with open(self.data_path, "r") as f:
            for line in f.readlines():
                line = self.__clean_line(line)
                if not line:
                    continue
                counts.update(line.split())

        vocab = {w: c for w, c in counts.items() if c >= self.min_freq}

        self.word2id = {w: i for i, w in enumerate(vocab.keys())}
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.sampling_table = self.__build_sampling_table(counts)

    def _load_and_tokenize(self):
        all_words = []
        offsets = [0]
        with open(self.data_path, "r") as f:
            for line in f.readlines():
                line = self.__clean_line(line)
                if not line:
                    continue

                all_words.extend(
                    [self.word2id[w] for w in line.split() if w in self.word2id]
                )
                offsets.append(len(all_words))

        self.corpus = np.array(all_words)
        self.offsets = np.array(offsets)
