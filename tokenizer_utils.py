"""データセットごとのトークナイザを meta.json から復元する共通窓口。

- char: STEP7の検証用（文字単位）
- bpe : STEP2のBedrock特化Byte-level BPE（Hugging Face tokenizers形式のtokenizer.json）
"""
from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.itos = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def decode_token(self, i: int) -> str:
        return self.itos[i]


class BPETokenizer:
    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(path))
        self.vocab_size = self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def decode_token(self, i: int) -> str:
        return self.tok.decode([i], skip_special_tokens=False)


def load_tokenizer(data_dir: str | Path) -> CharTokenizer | BPETokenizer:
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    if meta["type"] == "char":
        return CharTokenizer(meta["chars"])
    if meta["type"] == "bpe":
        return BPETokenizer(data_dir / meta["tokenizer_file"])
    raise ValueError(f"未知のトークナイザ種別: {meta['type']}")
