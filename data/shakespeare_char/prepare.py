"""STEP7用: Tiny Shakespeare を文字単位でトークン化して train.bin / val.bin を作る。

Bedrockデータを使わないのは、学習パイプラインのバグとデータ品質の問題を切り分けるため。
"""
import json
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
here = Path(__file__).parent

src = here / "input.txt"
if not src.exists():
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, src)

text = src.read_text(encoding="utf-8")
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in text], dtype=np.uint16)

n = int(len(ids) * 0.9)
ids[:n].tofile(here / "train.bin")
ids[n:].tofile(here / "val.bin")
(here / "meta.json").write_text(
    json.dumps({"type": "char", "vocab_size": len(chars), "chars": chars}, ensure_ascii=False),
    encoding="utf-8",
)
print(f"chars={len(text):,} vocab={len(chars)} train={n:,} val={len(ids) - n:,}")
