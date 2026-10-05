"""STEP9: コーパスを学習済みトークナイザでエンコードし、train.py が読む形式にする。

出力 (data/bedrock/): train.bin / val.bin (uint16), meta.json, tokenizer.json
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from train_tokenizer import split_documents

here = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="domain")
    a = ap.parse_args()

    src_tok = here / "tokenizer" / a.tokenizer / "tokenizer.json"
    shutil.copy(src_tok, here / "tokenizer.json")
    tok = Tokenizer.from_file(str(src_tok))
    vocab_size = tok.get_vocab_size()
    assert vocab_size < 2**16

    for split in ("train", "val"):
        docs = split_documents((here / "corpus" / f"{split}.txt").read_text(encoding="utf-8"))
        ids = [i for enc in tok.encode_batch(docs) for i in enc.ids]
        np.array(ids, dtype=np.uint16).tofile(here / f"{split}.bin")
        print(f"{split}: {len(docs):,} docs, {len(ids):,} tokens")

    (here / "meta.json").write_text(
        json.dumps({"type": "bpe", "vocab_size": vocab_size, "tokenizer_file": "tokenizer.json",
                    "tokenizer_kind": a.tokenizer}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
