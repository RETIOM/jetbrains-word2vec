from dataclasses import dataclass
import re
from collections import Counter


@dataclass(slots=True)
class WikiTextTokenizer:
    word2idx: dict[str, int]
    idx2word: dict[int, str]

    @staticmethod
    def build_vocab(data_path: str, min_count: int) -> WikiTextTokenizer:
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

        return WikiTextTokenizer(word2idx=word2idx, idx2word=idx2word)

    @staticmethod
    def _clean_line(line: str) -> str:
        if not (line := line.strip().lower()):
            return ""
        if re.match(r"^=.*=$", line) is not None:
            return ""

        line.replace("<unk>", "")
        return re.sub(r"[^a-z0-9]+", " ", line)

    def encode_line(self, line) -> list[int]:
        line = WikiTextTokenizer._clean_line(line)

        if not line:
            return []

        return [self.word2idx[w] for w in line.split() if w in self.word2idx]

    def encode(self, word):
        return self.word2idx[word]

    def decode(self, idx):
        return self.idx2word[idx]
