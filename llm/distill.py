"""蒸留（フェーズ1）: 先生モデルに依頼を解かせ、型チェック・BDS 実行テストを通った答えだけを教材として保存する。

    uv run python -m llm.distill --until 14:45            # 締め切り時刻の前に新しい依頼を始めない
    uv run python -m llm.distill --source templates --limit 20 --no_runtime   # お試し

依頼文: data/user/backtrans.jsonl（ユーザーのコードから逆算）+ data/distill/template_requests.jsonl。
先生: qwen3-coder:30b（Apache 2.0）+ RAG（型定義・早見表）+ 例示RAG + 型チェック/実行テストで最大3回の自己修正。

出力（すべて追記・再開可能。log にある ID は飛ばす）:
  data/distill/log.jsonl      … 全試行の記録（合否・ラウンド数・時間・残ったエラー）
  data/distill/answers.jsonl  … 合格した答え {id, request, scripts, output, rounds}
  data/distill/repairs.jsonl  … 自己修正の途中経過 {id, request, scripts(誤), errors, fixed(最終の正解)}

本番テスト用の評価問題（split=test）とよく似た依頼は、評価が甘くならないように除外する。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from llm.fim import PROJECT_ROOT
from mcaddon.examples import _grams

DIR = PROJECT_ROOT / "data" / "distill"
SOURCES = {"user": PROJECT_ROOT / "data" / "user" / "backtrans.jsonl",
           "templates": DIR / "template_requests.jsonl"}
TASKS = PROJECT_ROOT / "data" / "eval" / "user_tasks.jsonl"
OLD_API = [r"worldInitialize", r"world\.events\.", r"\bchatSend\b", r"executeCommand", r"runCommandAsync"]
LOCAL_IMPORT_RE = re.compile(r"""from\s+["'](?!@minecraft/)([^"']+)["']""")
LEAK_THRESHOLD = 0.35


def _cos(a, b) -> float:
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def test_similarity(requests: list[str]) -> list[float]:
    """各依頼と、本番テスト問題の指示文との最大類似度（文字バイグラムのコサイン）。"""
    tests = [_grams(json.loads(l)["instruction"]) for l in TASKS.read_text(encoding="utf-8").splitlines()
             if l and json.loads(l).get("split") == "test"]
    return [max((_cos(_grams(r), t) for t in tests), default=0.0) for r in requests]


def load_requests(sources: list[str]) -> list[dict]:
    rows = []
    for s in sources:
        for l in SOURCES[s].read_text(encoding="utf-8").splitlines():
            if l:
                r = json.loads(l)
                rows.append({"id": r["id"], "request": r["request"], "source": s})
    sims = test_similarity([r["request"] for r in rows])
    return [{**r, "test_sim": round(s, 3)} for r, s in zip(rows, sims)]


def self_contained(scripts: dict[str, str]) -> bool:
    """相対 import は、答えの中にあるファイルを指すものだけ（存在しない自作モジュールを使わない）。"""
    names = {p.removeprefix("scripts/") for p in scripts}
    for path, code in scripts.items():
        base = Path(path).parent
        for m in LOCAL_IMPORT_RE.finditer(code):
            target = (base / m.group(1)).as_posix()
            target = re.sub(r"(^|/)\./", r"\1", target).removeprefix("scripts/")
            if target not in names and target + ".js" not in names and target + ".ts" not in names:
                return False
    return True


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="qwen3-coder:30b")
    ap.add_argument("--source", nargs="+", choices=list(SOURCES), default=["templates", "user"])
    ap.add_argument("--limit", type=int, default=None, help="今回解く件数の上限")
    ap.add_argument("--until", default=None, help="HH:MM。この時刻の --margin 分前を過ぎたら新しい依頼を始めない")
    ap.add_argument("--margin", type=float, default=4.0, help="1件にかかる時間の目安（分）")
    ap.add_argument("--no_runtime", action="store_true", help="BDS の実行テストを省く（型チェックのみ）")
    ap.add_argument("--dry_run", action="store_true", help="対象件数と除外件数を表示するだけ")
    a = ap.parse_args()

    DIR.mkdir(parents=True, exist_ok=True)
    log_path, ans_path, rep_path = DIR / "log.jsonl", DIR / "answers.jsonl", DIR / "repairs.jsonl"
    done = set()
    if log_path.exists():
        done = {json.loads(l)["id"] for l in log_path.read_text(encoding="utf-8").splitlines() if l}
    reqs = load_requests(a.source)
    leak = [r for r in reqs if r["test_sim"] >= LEAK_THRESHOLD]
    todo = [r for r in reqs if r["test_sim"] < LEAK_THRESHOLD and r["id"] not in done]
    print(f"依頼 {len(reqs)} 件（テスト問題に似ていて除外 {len(leak)}、済み {len(done)}）→ 残り {len(todo)} 件", flush=True)
    if a.dry_run:
        for r in sorted(leak, key=lambda r: -r["test_sim"])[:15]:
            print(f"  除外 {r['test_sim']:.2f} {r['request'][:70]}")
        return
    if a.limit:
        todo = todo[:a.limit]

    deadline = None
    if a.until:
        deadline = datetime.combine(datetime.now().date(), datetime.strptime(a.until, "%H:%M").time())
        if deadline < datetime.now():
            deadline += timedelta(days=1)
        deadline -= timedelta(minutes=a.margin)

    from llm.eval_user import extract
    from mcaddon.script_gen import generate

    n_ok = 0
    for i, r in enumerate(todo, 1):
        if deadline and datetime.now() >= deadline:
            print(f"締め切り（{a.until}）が近いので終了します。", flush=True)
            break
        t0 = time.time()
        pack = r["id"].split(":")[0] if r["source"] == "user" else None
        try:
            res = generate(a.teacher, r["request"], extract, runtime=not a.no_runtime,
                           exclude_packs={pack} if pack else None)
        except Exception as e:  # Ollama のタイムアウトなど。記録して次へ
            _append(log_path, {"id": r["id"], "ok": False, "error": f"{type(e).__name__}: {e}"[:300],
                               "sec": round(time.time() - t0)})
            print(f"[{i}/{len(todo)}] {r['id']} 例外: {e}", flush=True)
            continue
        code = "\n".join(res.scripts.values())
        reasons = []
        if not res.scripts:
            reasons.append("スクリプトなし")
        if res.errors:
            reasons.append("エラーが残った")
        if any(re.search(p, code) for p in OLD_API):
            reasons.append("古いAPI")
        if res.scripts and not self_contained(res.scripts):
            reasons.append("存在しないファイルを import")
        ok = not reasons
        _append(log_path, {"id": r["id"], "source": r["source"], "ok": ok, "reasons": reasons, "rounds": res.rounds,
                           "sec": round(time.time() - t0), "errors": res.errors[:5], "runtime": not a.no_runtime,
                           **({} if ok else {"scripts": res.scripts, "jsons": res.jsons})})  # 不合格の最終コードは分析用
        if ok:
            n_ok += 1
            _append(ans_path, {"id": r["id"], "request": r["request"], "scripts": res.scripts, "jsons": res.jsons,
                               "output": res.output, "rounds": res.rounds, "runtime": not a.no_runtime})
            # 途中で失敗したコード + エラー → 最終の正解、を修正の教材にする
            for scripts, jsons, errors in res.attempts[:-1]:
                if (scripts or jsons) and errors:
                    _append(rep_path, {"id": r["id"], "request": r["request"], "scripts": scripts, "jsons": jsons,
                                       "errors": errors[:12], "fixed": res.scripts, "fixed_jsons": res.jsons})
        print(f"[{i}/{len(todo)}] {r['id'][:50]:50s} {'OK ' if ok else 'NG '} 修正{res.rounds}回 "
              f"{time.time() - t0:.0f}秒 {' '.join(reasons)}", flush=True)
    print(f"今回の合格 {n_ok} 件", flush=True)


if __name__ == "__main__":
    main()
