from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence


PAD = "<pad>"
UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"
SPECIALS: Sequence[str] = (PAD, UNK, BOS, EOS)


class CharTokenizer:
    """Fixed-size character vocabulary with UNK handling."""

    def __init__(self, id_to_char: List[str]):
        self.id_to_char = list(id_to_char)
        self.char_to_id = {c: i for i, c in enumerate(self.id_to_char)}
        self.pad_id = self.char_to_id["<pad>"]
        self.unk_id = self.char_to_id["<unk>"]
        self.bos_id = self.char_to_id[BOS]
        self.eos_id = self.char_to_id[EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_char)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids if 0 <= i < len(self.id_to_char))

    @staticmethod
    def build_from_text(
        text_stream: Iterator[str],
        max_vocab_size: int,
    ) -> "CharTokenizer":
        counter: Counter[str] = Counter()
        for chunk in text_stream:
            counter.update(chunk)
        most_common = [c for c, _ in counter.most_common(max(1, max_vocab_size - len(SPECIALS)))]
        id_to_char: List[str] = list(SPECIALS)
        for ch in most_common:
            if ch in id_to_char:
                continue
            id_to_char.append(ch)
            if len(id_to_char) >= max_vocab_size:
                break
        return CharTokenizer(id_to_char)

    def save(self, path: Path) -> None:
        payload = {"id_to_char": self.id_to_char, "version": 1}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: Path) -> "CharTokenizer":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return CharTokenizer(raw["id_to_char"])
