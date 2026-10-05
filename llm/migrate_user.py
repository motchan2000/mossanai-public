"""ユーザーの旧APIコード（newNSPVE の 1.18 系など、型チェックNGの単位）を 2.x に移行させ、移行の教材を作る。

教師: ローカルの qwen3-coder + RAG + 型チェックによる自己修正（mcaddon.script_gen）。
2.10.0 の型チェックを通ったものだけを data/user/migrated.jsonl に追記する（中断・再開可）。

    uv run python -m llm.migrate_user [--model qwen3-coder:30b] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from bedrock_tools.script_check import check_scripts
from llm.eval_user import extract
from llm.fim import PROJECT_ROOT
from mcaddon.script_gen import generate

UNITS = PROJECT_ROOT / "data" / "user" / "units.jsonl"
OUT = PROJECT_ROOT / "data" / "user" / "migrated.jsonl"
REQUEST = ("このスクリプトを @minecraft/server 2.x の安定版で動くように直して。処理の内容はそのままで、"
           "削除されたAPI（runCommandAsync など）は置き換えて。ファイル名は {file} のままで。\n\n"
           "```js\n{code}\n```")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-coder:30b")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    units = [json.loads(l) for l in UNITS.read_text(encoding="utf-8").splitlines() if l]
    todo = [u for u in units if not u["tsc_ok"]]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l}
    todo = [u for u in todo if f"{u['pack']}:{u['file']}:{u['start']}" not in done][: a.limit]
    print(f"{len(todo)} units to migrate", flush=True)
    t0, ok = time.time(), 0
    with open(OUT, "a", encoding="utf-8", newline="\n") as f:
        for n, u in enumerate(todo, 1):
            uid = f"{u['pack']}:{u['file']}:{u['start']}"
            original = (u["imports"] + "\n\n" if u["imports"] else "") + u["code"]
            before = [str(i) for i in check_scripts({u["file"]: original})]
            try:
                res = generate(a.model, REQUEST.format(file=u["file"], code=original), extract, max_repair=2)
            except Exception as e:
                print(f"  失敗 {uid}: {e}", flush=True)
                continue
            # 1ファイル分のスクリプトだけを採用（ファイル名が違っても中身が1つなら採用）
            scripts = res.scripts
            code = scripts.get(u["file"]) or (next(iter(scripts.values())) if len(scripts) == 1 else None)
            passed = bool(code) and not check_scripts({u["file"]: code})
            if passed:
                ok += 1
                f.write(json.dumps({"id": uid, "file": u["file"], "before": original, "before_errors": before,
                                    "after": code}, ensure_ascii=False) + "\n")
                f.flush()
            rate = (time.time() - t0) / n
            print(f"[{n}/{len(todo)}] {'OK' if passed else 'NG'} 合格{ok} {rate:.0f}s/件 "
                  f"残り約{rate * (len(todo) - n) / 60:.0f}分 {uid}", flush=True)


if __name__ == "__main__":
    main()
