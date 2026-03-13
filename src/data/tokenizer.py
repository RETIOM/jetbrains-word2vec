from dataclasses import dataclass
import re
from collections import Counter
import numpy as np
import os
from pathlib import Path


@dataclass(slots=True)
class Tokenizer:
    word2idx: dict[str, int]
    idx2word: dict[int, str]

    def encode(self, line):
        return [
            i
            for w in line.split()
            if (i := self.word2idx.get(w.strip().lower())) is not None
        ]

    def decode(self, idx):
        return self.idx2word[idx]

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_dir = Path(save_dir)

        np.savez(
            save_dir / "tokenizer.npz", word2idx=self.word2idx, idx2word=self.idx2word
        )

    @classmethod
    def from_vocab(cls, *args, **kwargs):
        pass

    @classmethod
    def from_file(cls, file_path: str):
        data = np.load(file_path, allow_pickle=True)

        word2idx = data["word2idx"].item()
        idx2word = data["idx2word"].item()

        return cls(word2idx=word2idx, idx2word=idx2word)


@dataclass(slots=True)
class WikiTextTokenizer(Tokenizer):
    @classmethod
    def from_vocab(cls, data_path: str, min_count: int) -> WikiTextTokenizer:
        counts: Counter = Counter()

        with open(data_path, "r") as f:
            for line in f.readlines():
                line = WikiTextTokenizer._clean_line(line)
                if not line:
                    continue
                counts.update(line.split())

        vocab = {w: c for w, c in counts.items() if c >= min_count}

        word2idx = {w: i for i, w in enumerate(vocab.keys())}
        idx2word = {i: w for w, i in word2idx.items()}

        return cls(word2idx=word2idx, idx2word=idx2word)

    @staticmethod
    def _clean_line(line: str) -> str:
        if not (line := line.strip().lower()):
            return ""
        if re.match(r"^=.*=$", line) is not None:
            return ""

        line = line.replace("<unk>", "")
        return re.sub(r"[^a-z0-9]+", " ", line)

    def encode(self, line) -> list[int]:
        line = WikiTextTokenizer._clean_line(line)
        return super().encode(line)

    @property
    def vocab_size(self):
        return len(self.word2idx)
