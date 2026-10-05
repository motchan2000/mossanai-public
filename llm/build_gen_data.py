"""「日本語の指示 → アドオンファイル生成」データセットの作成（CLI生成用）。

entity（+spawn_rules, loot_table）/ item / block の合成データを作り、
全ファイルが bedrock_tools の検証でエラー0件のものだけを採用する。

出力: data/sft/gen_train.jsonl / gen_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from bedrock_tools.check import check_file
from bedrock_tools.jsonc import dump
from bedrock_tools.schema_validator import SchemaValidator
from llm.chat import chat_completion, chat_prompt
from llm.fim import PROJECT_ROOT
from llm.synth_entity import answer_text, sample as entity_sample
from llm.synth_item_block import block_sample, item_sample

FILE_RE = re.compile(r"^### (\S+)\n```(?:json)?\n(.*?)\n```", re.S | re.M)


def parse_answer(text: str) -> list[tuple[str, str]]:
    """モデル出力から (パス, 本文) を取り出す。"""
    text = text.replace("\r\n", "\n")
    return [(m.group(1), m.group(2)) for m in FILE_RE.finditer(text)]


def make(rng: random.Random) -> tuple[str, list[tuple[str, dict]], str]:
    r = rng.random()
    if r < 0.6:
        ins, files, s = entity_sample(rng)
        return ins, files, f"entity_{s.archetype}"
    if r < 0.8:
        ins, files = item_sample(rng)
        return ins, files, "item"
    ins, files = block_sample(rng)
    return ins, files, "block"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=2500)
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--splits", nargs="+", default=["train"],
                    help="作るsplit。valは評価の基準なので、モデル間で比較するときは作り直さない")
    a = ap.parse_args()

    validator = SchemaValidator()
    tmp_dir = PROJECT_ROOT / "data" / "sft" / "_tmp_gen"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for split, n, seed in (("train", a.n_train, 10), ("val", a.n_val, 99)):
        if split not in a.splits:
            continue
        rng = random.Random(seed)
        rows, kinds, rejected = [], Counter(), 0
        while len(rows) < n:
            ins, files, kind = make(rng)
            ok = True
            for path, doc in files:
                tmp = tmp_dir / Path(path).name
                tmp.write_text(dump(doc), encoding="utf-8")
                if check_file(tmp, validator).errors:
                    ok = False
            if not ok:
                rejected += 1
                continue
            rows.append({"prompt": chat_prompt(ins), "completion": chat_completion(answer_text(files)),
                         "kind": kind, "instruction": ins})
            kinds[kind] += 1
        out = PROJECT_ROOT / "data" / "sft" / f"gen_{split}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(out, len(rows), "rejected", rejected, dict(kinds))
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()


if __name__ == "__main__":
    main()
