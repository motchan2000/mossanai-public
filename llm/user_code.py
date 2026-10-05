"""ユーザーの自作パック（許可済み）のスクリプトを、学習用の単位に分割する。

単位 = トップレベルの関数・クラス・定数・イベント購読（隣接する短い宣言はまとめる）。
各単位にはファイルの import 文を付け、型チェックの合否も記録する（2.10.0 で通るものをお手本として優先）。

    uv run python -m llm.user_code   →  data/user/units.jsonl
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from bedrock_tools.script_check import CHECK_DIR, check_scripts
from llm.fim import PROJECT_ROOT

MOJANG = Path(os.environ.get("APPDATA", "")) / "Minecraft Bedrock/Users/Shared/games/com.mojang"
# ユーザーが自分のものと確認したパック（2026-10-02）
PACKS = [
    MOJANG / "development_behavior_packs" / "RandomEquipment BP",
    MOJANG / "development_behavior_packs" / "RestaurantEmpire BP",
    MOJANG / "development_behavior_packs" / "LargeStructure BP",
    MOJANG / "development_behavior_packs" / "ScriptEventSuggest BP",
    MOJANG / "development_behavior_packs" / "MyCustomAddon",
    MOJANG / "development_behavior_packs" / "pve BP",
    MOJANG / "behavior_packs" / "newNSPVE",
    MOJANG / "behavior_packs" / "FactoryEmp",
]
MIN_LINES, MAX_LINES = 6, 160
OUT = PROJECT_ROOT / "data" / "user" / "units.jsonl"


def parse_files(files: list[Path]) -> list[dict]:
    res = subprocess.run(["node", str(CHECK_DIR / "chunk.mjs"), *map(str, files)], capture_output=True,
                         text=True, encoding="utf-8", check=True, cwd=CHECK_DIR)
    return json.loads(res.stdout)


def group_units(units: list[dict]) -> list[dict]:
    """短い宣言（定数など）は隣とまとめ、長すぎるものは除外する。"""
    groups, cur = [], []
    for u in units:
        n = u["end"] - u["start"] + 1
        if n > MAX_LINES:
            if cur:
                groups.append(cur)
                cur = []
            continue
        cur.append(u)
        size = sum(x["end"] - x["start"] + 1 for x in cur)
        if size >= MIN_LINES and u["kind"] in ("FunctionDeclaration", "ClassDeclaration", "ExpressionStatement") \
                or size >= 40:
            groups.append(cur)
            cur = []
    if cur and sum(x["end"] - x["start"] + 1 for x in cur) >= MIN_LINES:
        groups.append(cur)
    return groups


REL_IMPORT_RE = re.compile(r"""^\s*import\s.*from\s+["']\.\.?/.*$""", re.M)


def standalone_code(u: dict) -> str | None:
    """単体（プロジェクト内の他ファイルなし）で型チェックを通るなら、そのコード（相対 import を除いたもの）。"""
    imports = REL_IMPORT_RE.sub("", u["imports"]).strip()
    code = (imports + "\n\n" if imports else "") + u["code"]
    return code if not check_scripts({u["file"]: code}) else None


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for pack in PACKS:
        files = sorted(p for p in pack.rglob("*") if p.suffix in (".js", ".ts") and "node_modules" not in p.parts)
        if not files:
            continue
        # パック単位で型チェック（ファイル間の import を解決するため）
        rels = {p: p.relative_to(pack).as_posix() for p in files}
        issues = check_scripts({rels[p]: p.read_text(encoding="utf-8", errors="replace") for p in files})
        bad_lines: dict[str, set[int]] = {}
        for i in issues:
            bad_lines.setdefault(i.file, set()).add(i.line)
        for parsed in parse_files(files):
            path = Path(parsed["file"])
            rel = rels[path]
            for g in group_units(parsed["units"]):
                start, end = g[0]["start"], g[-1]["end"]
                errs = sorted(l for l in bad_lines.get(rel, ()) if start <= l <= end)
                row = {
                    "pack": pack.name, "file": rel, "start": start, "end": end,
                    "names": [u["name"] for u in g if u["name"]],
                    "imports": parsed["imports"],
                    "code": "\n".join(u["text"].rstrip() for u in g).strip("\n") + "\n",
                    "tsc_ok": not errs, "error_lines": errs,
                }
                # 単体で完結するか（「依頼 → 完成コード」の教材・お手本には自己完結が必要。v3 の失敗から）
                row["standalone"] = standalone_code(row) if row["tsc_ok"] else None
                rows.append(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter

    print(OUT, len(rows), "units")
    for pack, n in Counter(r["pack"] for r in rows).items():
        ok = sum(1 for r in rows if r["pack"] == pack and r["tsc_ok"])
        sa = sum(1 for r in rows if r["pack"] == pack and r["standalone"])
        print(f"  {pack}: {n} units（型チェックOK {ok}、単体で完結 {sa}）")


if __name__ == "__main__":
    main()
