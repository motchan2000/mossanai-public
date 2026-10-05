"""STEP2: Bedrock特化 Byte-level BPE トークナイザの学習と評価。

2方式を学習して比較する:
  standard … 設計図どおりの ByteLevelBPETokenizer（GPT-2と同じ正規表現で事前分割）
             → 記号と英字が必ず分割されるため "minecraft:entity" は原理的に1トークンにならない
  domain   … JSON文字列リテラル / 改行+インデント を1つの事前分割単位にする独自方式
             → "minecraft:behavior.melee_attack" のような頻出キーを丸ごと1トークンに学習できる

完了基準（設計図STEP2）:
  1. val全ドキュメントで encode→decode が完全一致
  2. GPT-2 tokenizerより同一テキストのトークン数が20〜40%少ない
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from build_corpus import ALL_TAGS

here = Path(__file__).parent

BASE_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
TAG_SPECIALS = [f"<{t}>" for t in ALL_TAGS] + [f"</{t}>" for t in ALL_TAGS]

# domain方式の事前分割。上から順に優先。
DOMAIN_PATTERN = (
    r"\n[ \t]*"                            # 改行 + インデント（JSONのネスト深さごとに1トークン化できる）
    r'| ?"(?:[^"\\\n]|\\.){0,64}"'          # 短いJSON文字列リテラル（キー・識別子）を丸ごと
    r"| ?[A-Za-z_$][A-Za-z0-9_$]*"           # 識別子・英単語
    r"| ?[0-9]+(?:\.[0-9]+)?"                # 数値
    r'| ?[^\sA-Za-z0-9_$"]+'                 # 記号列（ "}," や "=>" など）
    r"|[ \t]+"                              # 残りの空白
    r'|"'                                   # 長すぎる文字列の開き引用符
)


def build(kind: str) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    if kind == "standard":
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    else:
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(DOMAIN_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
    tok.decoder = decoders.ByteLevel()
    return tok


def train(kind: str, files: list[str], vocab_size: int) -> Tokenizer:
    tok = build(kind)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=BASE_SPECIALS + TAG_SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train(files, trainer)
    return tok


DOC_RE = re.compile(r"<(%s)>.*?</\1>\n" % "|".join(ALL_TAGS), re.S)


def split_documents(text: str) -> list[str]:
    """'<TAG>...</TAG>\n' 単位に分割（本文中に改行があるので行単位では切れない）。"""
    return [m.group(0) for m in DOC_RE.finditer(text)]


def evaluate(tok: Tokenizer, docs: list[str], gpt2) -> dict:
    n_ours = n_gpt2 = n_chars = mismatches = 0
    for d in docs:
        ids = tok.encode(d).ids
        if tok.decode(ids, skip_special_tokens=False) != d:
            mismatches += 1
        n_ours += len(ids)
        n_gpt2 += len(gpt2.encode(d, allowed_special=set()))
        n_chars += len(d)
    return {
        "tokens": n_ours,
        "gpt2_tokens": n_gpt2,
        "reduction_vs_gpt2": round(1 - n_ours / n_gpt2, 4),
        "chars_per_token": round(n_chars / n_ours, 3),
        "roundtrip_mismatches": mismatches,
    }


PROBE = ['"format_version"', '"minecraft:entity"', '"components"', '"minecraft:behavior.melee_attack"',
         '"minecraft:on_target_acquired"', '"minecraft:health"', "query.is_baby", "variable.attack_time",
         "math.sin", "q.", "v.", "world.afterEvents", "@minecraft/server"]


def main() -> None:
    import tiktoken

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=here / "corpus")
    ap.add_argument("--vocab_size", type=int, default=12000)
    ap.add_argument("--kinds", nargs="+", default=["standard", "domain"])
    a = ap.parse_args()

    gpt2 = tiktoken.get_encoding("gpt2")
    val_docs = split_documents((a.corpus / "val.txt").read_text(encoding="utf-8"))
    report = {}
    for kind in a.kinds:
        tok = train(kind, [str(a.corpus / "train.txt")], a.vocab_size)
        out = here / "tokenizer" / kind
        out.mkdir(parents=True, exist_ok=True)
        tok.save(str(out / "tokenizer.json"))
        res = evaluate(tok, val_docs, gpt2)
        res["probe"] = {p: len(tok.encode(p).ids) for p in PROBE}
        report[kind] = res
        print(f"[{kind}] {json.dumps(res, ensure_ascii=False)}")
    report["gpt2_probe"] = {p: len(gpt2.encode(p)) for p in PROBE}
    (here / "tokenizer" / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
