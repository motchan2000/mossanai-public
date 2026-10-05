"""「私がしそうな問題」（data/eval/user_tasks.jsonl）での評価。

    uv run python -m llm.eval_user --models mossanai-v2 qwen3-coder:30b

各問題の判定:
  code    … 求められた種類の出力がある（script / json）
  tsc     … 出力スクリプトが型チェック（@minecraft/server 2.10.0）を通る
  api     … require の正規表現（使うべきAPI）をすべて含む
  no_old  … forbid（削除済み・beta専用のAPI）を含まない
  json    … JSONがパースでき、検証エラー0件で、求められた種類（block / item / entity）がある
  PASS    … 該当する判定がすべて合格
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

from bedrock_tools.check import check_file
from bedrock_tools.jsonc import parse_lenient
from bedrock_tools.schema_validator import SchemaValidator, detect_type
from bedrock_tools.script_check import check_scripts
from llm.fim import PROJECT_ROOT
from mcaddon.ollama import chat
from mcaddon.script_gen import generate

BLOCK_RE = re.compile(r"(?:^###\s*(?P<path>\S+)\s*\n)?```(?P<lang>[\w+-]*)\n(?P<body>.*?)```", re.S | re.M)
PATH_COMMENT_RE = re.compile(r"^\s*//\s*(?P<path>[\w./-]+\.(?:js|ts))\s*$", re.M)
JSON_KIND = {"minecraft:block": "block", "minecraft:item": "item", "minecraft:entity": "entity"}
# 問題の "from" → data/user/units.jsonl の pack 名
PACK_DIRS = {"RandomEquipment": "RandomEquipment BP", "RestaurantEmpire": "RestaurantEmpire BP",
             "LargeStructure": "LargeStructure BP", "ScriptEventSuggest": "ScriptEventSuggest BP", "pve": "pve BP"}


def extract(text: str, lang_default: str) -> tuple[dict[str, str], list[str]]:
    """出力から (スクリプト {path: code}, JSON本文のリスト) を取り出す。"""
    scripts, jsons = {}, []
    for i, m in enumerate(BLOCK_RE.finditer(text.replace("\r\n", "\n"))):
        body, lang, path = m["body"], m["lang"].lower(), m["path"]
        if lang == "json" or (not lang and body.lstrip().startswith("{")):
            jsons.append(body)
            continue
        if lang not in ("", "js", "javascript", "ts", "typescript", "mjs"):
            continue
        # 言語指定なしのブロックはフォルダ構成図などのこともあるので、JSらしいものだけ採用
        if not lang and not re.search(r"\b(import|export|function|const|let|class)\b|=>", body):
            continue
        if not path:
            pc = PATH_COMMENT_RE.search(body.split("\n", 1)[0])
            path = pc["path"] if pc else None
        ext = ".ts" if lang in ("ts", "typescript") or lang_default == "ts" else ".js"
        if not path or not path.endswith((".js", ".ts")):
            path = f"scripts/main{'' if i == 0 else i}{ext}"
        path = path.removeprefix("BP/").lstrip("/")
        if not path.startswith("scripts/"):
            path = "scripts/" + path.split("scripts/")[-1]
        scripts[path] = body
    return scripts, jsons


CONTEXT = ("【前提】Minecraft 統合版（Bedrock Edition）のアドオン開発です。スクリプトは Script API "
           "（@minecraft/server 2.x 安定版・JavaScript）、JSON はビヘイビアパック/リソースパックの形式で書いてください。\n\n")


def build_prompt(task: dict, context: bool = False) -> str:
    """context=True は汎用モデル比較用（mossanai は統合版専用なので前提なしで使う想定）。"""
    p = (CONTEXT if context else "") + task["instruction"]
    if task.get("input_code"):
        p += "\n\n```js\n" + task["input_code"] + "```"
    return p


def judge(task: dict, out: str, validator: SchemaValidator, tmp: Path, runtime: bool = False) -> dict:
    scripts, jsons = extract(out, task.get("lang", "js"))
    kind = task["kind"]
    need_script = kind in ("script", "fix", "mixed")
    need_json = kind in ("json", "mixed")
    r: dict = {"code": (not need_script or bool(scripts)) and (not need_json or bool(jsons))}

    if need_script:
        issues = check_scripts(scripts) if scripts else None
        r["tsc"] = issues == []
        r["tsc_errors"] = [str(i) for i in (issues or [])][:5]
        code = "\n".join(scripts.values())
        miss = [p for p in task.get("require", []) if not re.search(p, code)]
        r["api"] = bool(scripts) and not miss
        r["api_missing"] = miss
        r["no_old"] = not any(re.search(p, code) for p in task.get("forbid", []))
    if need_json:
        kinds, errors = set(), []
        for body in jsons:
            doc, err = parse_lenient(body)
            if err:
                errors.append(err)
                continue
            for k, v in JSON_KIND.items():
                if isinstance(doc, dict) and k in doc:
                    kinds.add(v)
            f = tmp / ("manifest.json" if isinstance(doc, dict) and "header" in doc else "x.json")
            f.write_text(body, encoding="utf-8")
            errors += check_file(f, validator).errors
        r["json"] = not errors and set(task.get("json_kinds", [])) <= kinds
        r["json_errors"] = errors[:5]
    if runtime and need_script and r.get("tsc"):
        from bedrock_tools.runtime_check import pack_json_files, run_pack

        # JSON（ブロック・アイテム等）もパックに入れて読み込ませる（カスタムコンポーネントの紐付け確認のため）
        rt = run_pack(scripts, json_files=pack_json_files(jsons))
        r["runtime"] = rt.ok
        r["runtime_errors"] = rt.errors[:3]
    r["PASS"] = all(v for k, v in r.items() if k in ("code", "tsc", "api", "no_old", "json", "runtime"))
    return r


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tasks", type=Path, default=PROJECT_ROOT / "data" / "eval" / "user_tasks.jsonl")
    ap.add_argument("--only", nargs="*", default=None, help="問題IDで絞り込み")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev",
                    help="dev=開発用（調整に使ってよい）/ test=本番テスト用（最終評価のみ）")
    ap.add_argument("--context", action="store_true", help="統合版である前提を問題文に付ける（汎用モデル比較用）")
    ap.add_argument("--rejudge", action="store_true", help="モデルを呼ばず、保存済みの出力を採点し直す")
    ap.add_argument("--rag", action="store_true", help="公式型定義を参照させ、型チェックで自己修正する（mcaddon.script_gen）")
    ap.add_argument("--compact", action="store_true", help="RAGの文脈を mossanai 学習時と同じ小さいものにする")
    ap.add_argument("--no_examples", action="store_true", help="例示RAG（お手本）を使わない")
    ap.add_argument("--runtime", action="store_true", help="型チェックOKのスクリプトを BDS で実際に読み込ませて実行時エラーも判定")
    a = ap.parse_args()

    tasks = [json.loads(l) for l in a.tasks.read_text(encoding="utf-8").splitlines() if l]
    if a.split != "all":
        tasks = [t for t in tasks if t.get("split", "dev") == a.split]
    if a.only:
        tasks = [t for t in tasks if t["id"] in a.only]
    validator = SchemaValidator()
    tmp = Path(tempfile.mkdtemp(prefix="eval_user_"))
    out_dir = PROJECT_ROOT / "out" / "eval"
    for model in a.models:
        safe = model.replace(":", "_").replace("/", "_") + ("_test" if a.split == "test" else "") \
            + ("_ctx" if a.context else "") + ("_rag" if a.rag else "") + ("c" if a.rag and a.compact else "")
        saved = {}
        if a.rejudge:
            saved = {json.loads(l)["id"]: json.loads(l)["output"]
                     for l in (out_dir / f"{safe}_user.jsonl").read_text(encoding="utf-8").splitlines() if l}
        rows, t0 = [], time.time()
        for t in tasks:
            rounds = None
            if a.rejudge:
                out = saved[t["id"]]
            elif a.rag:
                # 問題の題材になったパックのお手本は使わない（答えの丸写しを防ぎ、汎化を測る）
                res = generate(model, build_prompt(t), extract, lang=t.get("lang", "js"), compact=a.compact,
                               examples=not a.no_examples, exclude_packs={PACK_DIRS.get(t["from"], t["from"])},
                               runtime=a.runtime)
                rounds = res.rounds
                # 採点対象: 自己修正後の最終の JSON とスクリプト
                out = "\n\n".join([f"```json\n{j}\n```" for j in res.jsons] +
                                  [f"### {p}\n```{'ts' if p.endswith('.ts') else 'js'}\n{c}\n```"
                                   for p, c in res.scripts.items()])
            else:
                out = chat(model, build_prompt(t, a.context), temperature=0.0, max_tokens=2500, timeout=1800)
            r = judge(t, out, validator, tmp, runtime=a.runtime)
            rows.append({"id": t["id"], "kind": t["kind"], **r, "repair_rounds": rounds, "output": out})
            marks = " ".join(f"{k}={'○' if r[k] else '×'}" for k in ("code", "tsc", "api", "no_old", "json", "runtime")
                             if k in r)
            print(f"[{model}] {t['id']:20s} {'PASS' if r['PASS'] else 'fail'}  {marks}"
                  + (f"  未使用API={r['api_missing']}" if r.get("api_missing") else "")
                  + (f"  tsc: {r['tsc_errors'][0]}" if r.get("tsc_errors") else ""), flush=True)
        n = len(rows)
        summary = {"model": model, "context": a.context, "n": n, "pass": sum(r["PASS"] for r in rows) / n,
                   "minutes": round((time.time() - t0) / 60, 1)}
        for k in ("code", "tsc", "api", "no_old", "json", "runtime"):
            vals = [r[k] for r in rows if k in r]
            if vals:
                summary[k] = sum(vals) / len(vals)
        print(json.dumps(summary, ensure_ascii=False))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{safe}_user.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                                                    encoding="utf-8")
        (out_dir / f"{safe}_user.summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                            encoding="utf-8")


if __name__ == "__main__":
    main()
