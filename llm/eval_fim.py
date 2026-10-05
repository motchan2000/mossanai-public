"""FIM補完の評価（ベースモデル / 学習後モデルを同じ問題で比較）。

指標:
  exact  … 期待値と完全一致（前後の空白は無視）
  valid  … 補完後のファイルがJSONとしてパースでき、かつ検証エラー数が元ファイル以下
  latency… 1件あたりの生成時間（transformers bf16。実運用のllama.cppとは異なる参考値）

    uv run python -m llm.eval_fim [--adapter out/lora/xxx] [--limit 100]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from bedrock_tools.check import check_file
from bedrock_tools.jsonc import parse_lenient
from bedrock_tools.schema_validator import SchemaValidator
from llm.fim import DEFAULT_BASE, complete, fim_prompt, load
from llm.fim_tasks import default_eval_path, load_tasks


def error_count(text: str, validator: SchemaValidator, tmp: Path) -> int | None:
    """パース不能ならNone、そうでなければ検証エラー数。"""
    if parse_lenient(text)[1]:
        return None
    tmp.write_text(text, encoding="utf-8")
    return len(check_file(tmp, validator).errors)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_BASE))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tasks", type=Path, default=default_eval_path())
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--with_path", action="store_true", help="ファイルパスをプロンプトに含める")
    ap.add_argument("--out", type=Path, default=None, help="結果のjsonl出力先")
    a = ap.parse_args()

    tasks = load_tasks(a.tasks)[: a.limit]
    tok, model = load(a.model, a.adapter)
    validator = SchemaValidator()
    tmp = Path(a.tasks).parent / "_tmp_eval.json"

    rows, per = [], defaultdict(lambda: defaultdict(list))
    for t in tqdm(tasks, desc="FIM"):
        prompt = fim_prompt(t.prefix, t.suffix, t.path if a.with_path else None)
        c = complete(tok, model, prompt, max_new_tokens=64 if t.kind == "line" else 256,
                     stop_at_newline=t.kind == "line")
        exact = c.text.strip() == t.expected.strip()
        base_err = error_count(t.full(t.expected), validator, tmp)
        new_err = error_count(t.full(c.text), validator, tmp)
        valid = new_err is not None and (base_err is None or new_err <= base_err)
        group = t.path.split("/")[1]
        for key in (t.kind, f"{t.kind}:{group}"):
            per[key]["exact"].append(exact)
            per[key]["valid"].append(valid)
            per[key]["sec"].append(c.seconds)
        rows.append({"id": t.id, "kind": t.kind, "exact": exact, "valid": valid,
                     "sec": round(c.seconds, 3), "got": c.text, "expected": t.expected})
    tmp.unlink(missing_ok=True)

    summary = {}
    print(f"\nmodel={a.model} adapter={a.adapter} tasks={len(tasks)}")
    print(f"{'group':32s} {'n':>4s} {'exact':>7s} {'valid':>7s} {'p50 ms':>7s}")
    for key in sorted(per, key=lambda k: (":" in k, k)):
        m = per[key]
        n = len(m["exact"])
        row = {"n": n, "exact": sum(m["exact"]) / n, "valid": sum(m["valid"]) / n,
               "p50_ms": statistics.median(m["sec"]) * 1000}
        summary[key] = row
        print(f"{key:32s} {n:4d} {row['exact']:7.1%} {row['valid']:7.1%} {row['p50_ms']:7.0f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        a.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")


if __name__ == "__main__":
    main()
