"""「日本語の指示 → ファイル生成」の評価。

  files_ok  … 期待したファイル（パス）がすべて出力された
  valid     … 出力された全ファイルがパースでき、検証エラー0件
  value_acc … 正解JSONの葉の値（体力・ID・ダメージ等）のうち、出力の同じ位置に同じ値がある割合
              （指示どおりに作れているかの指標。パスが一致したファイル同士で比較）

    uv run python -m llm.eval_gen [--adapter out/lora/xxx/best] [--limit 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from bedrock_tools.check import check_file
from bedrock_tools.jsonc import parse_lenient
from bedrock_tools.schema_validator import SchemaValidator
from llm.build_gen_data import parse_answer
from llm.chat import IM_END, user_text
from llm.eval_repair import chat_generate
from llm.fim import DEFAULT_BASE, PROJECT_ROOT, load


def leaves(node, prefix=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, prefix + (k,))
    elif isinstance(node, list) and node and all(isinstance(x, (dict, list)) for x in node):
        for i, v in enumerate(node):
            yield from leaves(v, prefix + (i,))
    else:
        yield prefix, node


def get_path(node, path):
    for p in path:
        if isinstance(node, dict) and p in node:
            node = node[p]
        elif isinstance(node, list) and isinstance(p, int) and p < len(node):
            node = node[p]
        else:
            return KeyError
    return node


def value_accuracy(expected: dict, got) -> tuple[int, int]:
    pairs = list(leaves(expected))
    hit = sum(1 for path, v in pairs if get_path(got, path) == v)
    return hit, len(pairs)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_BASE))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "sft" / "gen_val.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ollama", default=None, help="Ollamaのモデル名（指定するとOllama経由で評価）")
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.data.read_text(encoding="utf-8").splitlines() if l][: a.limit]
    if a.ollama:
        from transformers import AutoTokenizer

        from mcaddon.ollama import chat

        tok, model = AutoTokenizer.from_pretrained(a.model), None
        generate = lambda prompt, n: chat(a.ollama, user_text(prompt), temperature=0.0, max_tokens=n)  # noqa: E731
    else:
        tok, model = load(a.model, a.adapter)
        generate = lambda prompt, n: chat_generate(tok, model, prompt, n)  # noqa: E731
    validator = SchemaValidator()
    tmp_dir = a.data.parent / "_tmp_eval_gen"
    tmp_dir.mkdir(exist_ok=True)

    per = defaultdict(lambda: defaultdict(float))
    results = []
    for r in tqdm(rows, desc="gen"):
        expected = {p: parse_lenient(t)[0] for p, t in parse_answer(r["completion"].removesuffix(IM_END))}
        n_tok = len(tok(r["completion"]).input_ids)
        out = generate(r["prompt"], int(n_tok * 1.3) + 64)
        got_files = dict(parse_answer(out))
        files_ok = set(expected) <= set(got_files)
        valid, hit, total, errors = bool(got_files), 0, 0, []
        for path, text in got_files.items():
            tmp = tmp_dir / Path(path).name
            tmp.write_text(text, encoding="utf-8")
            rep = check_file(tmp, validator)
            if rep.errors:
                valid = False
                errors += rep.errors[:2]
        for path, exp in expected.items():
            got_doc = parse_lenient(got_files[path])[0] if path in got_files else None
            h, t = value_accuracy(exp, got_doc)
            hit, total = hit + h, total + t
        for key in (r["kind"], "ALL"):
            per[key]["n"] += 1
            per[key]["files_ok"] += files_ok
            per[key]["valid"] += valid
            per[key]["hit"] += hit
            per[key]["total"] += total
        results.append({"kind": r["kind"], "instruction": r["instruction"], "files_ok": files_ok, "valid": valid,
                        "value_acc": hit / max(total, 1), "errors": errors, "output": out})
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    print(f"\nmodel={a.ollama or a.model} adapter={a.adapter}")
    print(f"{'kind':20s} {'n':>4s} {'files_ok':>9s} {'valid':>7s} {'value_acc':>10s}")
    summary = {}
    for k in sorted(per, key=lambda k: (k != "ALL", k)):
        m = per[k]
        n = int(m["n"])
        summary[k] = {"n": n, "files_ok": m["files_ok"] / n, "valid": m["valid"] / n,
                      "value_acc": m["hit"] / max(m["total"], 1)}
        s = summary[k]
        print(f"{k:20s} {n:4d} {s['files_ok']:9.1%} {s['valid']:7.1%} {s['value_acc']:10.1%}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results), encoding="utf-8")
        a.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
