"""「検証エラー → 修正」タスクの評価。

  fixed … 出力が検証ツールでエラー0件（パース可能）
  exact … 出力JSONが正解と同じ内容（整形の違いは無視）

    uv run python -m llm.eval_repair [--adapter out/lora/xxx/best] [--limit 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from bedrock_tools.check import check_file
from bedrock_tools.jsonc import parse_lenient
from bedrock_tools.schema_validator import SchemaValidator
from llm.chat import IM_END, extract_code_block, user_text
from llm.fim import DEFAULT_BASE, PROJECT_ROOT, load


@torch.no_grad()
def chat_generate(tok, model, prompt: str, max_new_tokens: int) -> str:
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    stop = [tok.convert_tokens_to_ids(IM_END), tok.convert_tokens_to_ids("<|endoftext|>")]
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=stop, pad_token_id=stop[1])
    return tok.decode(out[0, ids.size(1):], skip_special_tokens=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_BASE))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "sft" / "repair_val.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--ollama", default=None, help="Ollamaのモデル名（指定するとOllama経由で評価）")
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.data.read_text(encoding="utf-8").splitlines() if l][: a.limit]
    if a.ollama:
        from transformers import AutoTokenizer

        from mcaddon.ollama import chat

        tok = AutoTokenizer.from_pretrained(a.model)
        generate = lambda prompt, n: chat(a.ollama, user_text(prompt), temperature=0.0, max_tokens=n)  # noqa: E731
    else:
        tok, model = load(a.model, a.adapter)
        generate = lambda prompt, n: chat_generate(tok, model, prompt, n)  # noqa: E731
    validator = SchemaValidator()
    tmp_dir = a.data.parent / "_tmp_eval"
    tmp_dir.mkdir(exist_ok=True)

    per, results = defaultdict(lambda: [0, 0, 0]), []
    for r in tqdm(rows, desc="repair"):
        expected_text = extract_code_block(r["completion"].removesuffix(IM_END))
        n_tok = len(tok(expected_text).input_ids)
        out = generate(r["prompt"], int(n_tok * 1.3) + 64)
        got = extract_code_block(out)
        tmp = tmp_dir / ("manifest.json" if r["path"] == "manifest.json" else "x.json")
        tmp.write_text(got, encoding="utf-8")
        rep = check_file(tmp, validator)
        fixed = not rep.errors
        got_doc, err = parse_lenient(got)
        exact = err is None and got_doc == parse_lenient(expected_text)[0]
        for key in (r["kind"], "ALL"):
            per[key][0] += 1
            per[key][1] += fixed
            per[key][2] += exact
        results.append({"kind": r["kind"], "path": r["path"], "fixed": fixed, "exact": exact,
                        "errors": rep.errors[:3]})
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()

    print(f"\nmodel={a.ollama or a.model} adapter={a.adapter}")
    print(f"{'kind':30s} {'n':>4s} {'fixed':>7s} {'exact':>7s}")
    summary = {}
    for k in sorted(per, key=lambda k: (k != "ALL", k)):
        n, f, e = per[k]
        summary[k] = {"n": n, "fixed": f / n, "exact": e / n}
        print(f"{k:30s} {n:4d} {f / n:7.1%} {e / n:7.1%}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results), encoding="utf-8")
        a.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
