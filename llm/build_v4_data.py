"""mossanai-v4（7B, CLI 生成用）の教材を作る。プロンプトは実際に使うとき（mcaddon.script_gen, compact=True）と同じ形。

  gen      : 依頼 → 型チェック・実行テストを通ったコード
             （蒸留の合格答え data/distill/answers.jsonl + ユーザーの自己完結したコード）
  fix      : 自己修正の途中経過（data/distill/repairs.jsonl）+ 正しいコードを典型的な誤りで壊したもの → 正しいコード
  json_*   : v2 の JSON 生成・修正の教材から一部（JSON の力を落とさないため）

    uv run python -m llm.build_v4_data   # → data/sft/v4_{train,val}.jsonl

本番テスト問題（split=test）に似た依頼は除外する（llm.distill と同じ基準）。
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from bedrock_tools.jsonc import parse_lenient
from bedrock_tools.script_check import check_scripts
from llm.build_v3_data import MUTATIONS, is_val, normalize_request, unit_id
from llm.chat import chat_completion, chat_prompt
from llm.distill import LEAK_THRESHOLD, test_similarity
from llm.fim import PROJECT_ROOT
from mcaddon.script_gen import check_jsons, fix_prompt, format_scripts, gen_context, gen_prompt

SFT = PROJECT_ROOT / "data" / "sft"
DISTILL = PROJECT_ROOT / "data" / "distill"
UNITS = PROJECT_ROOT / "data" / "user" / "units.jsonl"
BACKTRANS = PROJECT_ROOT / "data" / "user" / "backtrans.jsonl"
JSON_FOLDER = (("minecraft:block", "blocks"), ("minecraft:item", "items"), ("minecraft:entity", "entities"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l] if path.exists() else []


def answer_jsons(a: dict) -> list[str] | None:
    """答えの JSON（ブロック・アイテム・エンティティ）。検証で問題があれば None（その答えは使わない）。"""
    from llm.eval_user import extract

    bodies = a["jsons"] if "jsons" in a else extract(a["output"], "js")[1]  # 古い記録は最初の出力から
    keep = []
    for body in bodies:
        doc, err = parse_lenient(body)
        if err or not isinstance(doc, dict):
            return None
        if any(k in doc for k, _ in JSON_FOLDER):
            keep.append(body.rstrip())  # manifest などは答えに含めない（パックの雛形は CLI 側で作る）
    return None if check_jsons(keep) else keep


def mutate(scripts: dict[str, str], rng: random.Random) -> tuple[dict[str, str], list[str], str] | None:
    """1か所だけ典型的な誤りを入れ、型チェックで検出できたものを返す。"""
    cands = [(p, m) for p, c in scripts.items() if p.endswith(".js") for m in MUTATIONS if re.search(m[1], c)]
    rng.shuffle(cands)
    for path, (name, pat, repl) in cands[:3]:
        broken = re.sub(pat, repl, scripts[path], count=1)
        if broken == scripts[path]:
            continue
        b = {**scripts, path: broken}
        errors = [str(i) for i in check_scripts(b)]
        if errors:
            return b, errors, name
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(4)
    rows: list[dict] = []
    kinds = Counter()

    def add(kind: str, key: str, user: str, answer: str) -> None:
        rows.append({"prompt": chat_prompt(user), "completion": chat_completion(answer), "kind": kind,
                     "split": "val" if is_val(key) else "train"})
        kinds[kind] += 1

    # ---- 生成の元: (キー, 依頼, スクリプト, JSON, 除外パック) ----
    sources = []
    bad_json = 0
    for a in _jsonl(DISTILL / "answers.jsonl"):
        jsons = answer_jsons(a)
        if jsons is None:
            bad_json += 1
            continue
        pack = a["id"].split(":")[0] if not a["id"].startswith("tpl:") else None
        sources.append((a["id"], a["request"], a["scripts"], jsons, pack))
    requests = {b["id"]: b["request"] for b in _jsonl(BACKTRANS)}
    for u in _jsonl(UNITS):
        uid = unit_id(u)
        req = normalize_request(requests.get(uid, "")) if uid in requests else None
        if u.get("standalone") and req:
            code = u["standalone"].replace("\r\n", "\n")
            sources.append((uid, req, {f"scripts/{Path(u['file']).name}": code}, [], u["pack"]))
    sims = test_similarity([s[1] for s in sources])
    leaked = sum(s >= LEAK_THRESHOLD for s in sims)
    sources = [s for s, sim in zip(sources, sims) if sim < LEAK_THRESHOLD]
    print(f"生成の元 {len(sources)} 件（JSON に誤りがあり除外 {bad_json}、テスト問題に似ていて除外 {leaked}）")

    for key, req, scripts, jsons, pack in sources:
        ctx = gen_context(req, compact=True, examples=True, exclude_packs={pack} if pack else None)
        add("gen", key, gen_prompt(ctx, req), format_scripts(scripts, jsons))
        m = mutate(scripts, rng)
        if m:
            broken, errors, name = m
            add(f"fix_{name}", key, fix_prompt(ctx, req, broken, errors, jsons), format_scripts(scripts, jsons))

    # ---- 蒸留の自己修正の途中経過 ----
    for r in _jsonl(DISTILL / "repairs.jsonl"):
        if test_similarity([r["request"]])[0] >= LEAK_THRESHOLD:
            continue
        pack = r["id"].split(":")[0] if not r["id"].startswith("tpl:") else None
        ctx = gen_context(r["request"], compact=True, examples=True, exclude_packs={pack} if pack else None)
        add("fix_teacher", r["id"], fix_prompt(ctx, r["request"], r["scripts"], r["errors"], r.get("jsons")),
            format_scripts(r["fixed"], r.get("fixed_jsons", r.get("jsons"))))

    # ---- JSON の力を保つ（v2 の教材から一部） ----
    for name, n in (("gen", 1500), ("repair", 800)):
        for split in ("train", "val"):
            src = _jsonl(SFT / f"{name}_{split}.jsonl")
            rng.shuffle(src)
            for r in src[: n if split == "train" else n // 20]:
                rows.append({"prompt": r["prompt"], "completion": r["completion"], "kind": f"json_{name}", "split": split})
                kinds[f"json_{name}"] += 1

    for split in ("train", "val"):
        part = [r for r in rows if r["split"] == split]
        rng.shuffle(part)
        path = SFT / f"v4_{split}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in part:
                f.write(json.dumps({k: v for k, v in r.items() if k != "split"}, ensure_ascii=False) + "\n")
        print(f"{path.name}: {len(part)}")
    print(dict(kinds))


if __name__ == "__main__":
    main()
