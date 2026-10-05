"""mossanai-v3 の Script API 系の教材を作る（ユーザーの自作パックのコード＝許可済み、が中心）。

  script_gen    : 依頼文（llm/backtranslate.py で逆算）→ ユーザーのコード。RAG の文脈付き（7割）/なし（3割）
  script_fim    : ユーザーのスクリプト（型チェックOKの単位）の穴埋め（VSCode 補完用）
  script_repair : 正しいコードに典型的な誤り（削除済みAPI、Player 判定漏れ等）を入れ、tsc のエラーから直させる
  user_json_fim : ユーザーの JSON（検証OK）の穴埋め

出力: data/sft/v3_{名前}_{train|val}.jsonl
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from bedrock_tools.check import check_file
from bedrock_tools.schema_validator import SchemaValidator
from bedrock_tools.script_check import check_scripts
from llm.build_fim_data import make_samples as json_fim_samples
from llm.chat import chat_completion, chat_prompt
from llm.fim import PROJECT_ROOT, fim_prompt
from llm.user_code import MOJANG, PACKS
from mcaddon.rag import CHEATSHEET, repair_hints, retrieve
from bedrock_tools.api_index import signature

SFT = PROJECT_ROOT / "data" / "sft"
UNITS = PROJECT_ROOT / "data" / "user" / "units.jsonl"
BACKTRANS = PROJECT_ROOT / "data" / "user" / "backtrans.jsonl"
EOT = "<|endoftext|>"
VAL_RATIO = 0.05


def is_val(key: str) -> bool:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < VAL_RATIO


def unit_id(u: dict) -> str:
    return f"{u['pack']}:{u['file']}:{u['start']}"


def compact_context(request: str, max_chars: int = 5000) -> str:
    """学習・推論で共通の、小さいモデル向けの RAG 文脈（早見表 + 関係する型定義を少し）。"""
    sigs = []
    total = 0
    for name in retrieve(request, max_chars=max_chars):
        s = signature(name, 2500)
        if s and total + len(s) <= max_chars:
            sigs.append(s)
            total += len(s)
    body = "\n\n".join(sigs)
    return f"{CHEATSHEET}\n# 関係しそうな API（@minecraft/server 2.10.0）\n```ts\n{body}\n```\n"


def script_answer(file: str, imports: str, code: str) -> str:
    lang = "ts" if file.endswith(".ts") else "js"
    body = (imports + "\n\n" if imports else "") + code.rstrip()
    return f"### {file}\n```{lang}\n{body}\n```"


# ---------------------------------------------------------------------------
# 教師（qwen3-coder）がときどき混ぜる中国語（簡体字）の対策
ZH_FIX = {"音效": "効果音", "设置": "設定", "实现": "実装", "获取": "取得", "显示": "表示", "处理": "処理"}
ZH_ONLY = set("么们这个设击对时后为过吗让说获实现发应该还没东边见长门问间关开车")


def normalize_request(text: str) -> str | None:
    for zh, ja in ZH_FIX.items():
        text = text.replace(zh, ja)
    return None if any(c in ZH_ONLY for c in text) else text


def build_script_gen(units: dict[str, dict], rng: random.Random) -> dict[str, list]:
    out = {"train": [], "val": []}
    dropped = 0
    for line in BACKTRANS.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        b = json.loads(line)
        u = units.get(b["id"])
        if not u:
            continue
        req = normalize_request(b["request"])
        if req is None:
            dropped += 1
            continue
        user = (f"{compact_context(req)}\n# 依頼\n{req}") if rng.random() < 0.7 else req
        split = "val" if is_val(u["file"]) else "train"
        out[split].append({"prompt": chat_prompt(user), "kind": "script_gen", "id": b["id"],
                           "completion": chat_completion(script_answer(u["file"], u["imports"], u["code"]))})
    print(f"  script_gen: 簡体字が残る依頼文 {dropped} 件を除外")
    return out


def build_script_fim(units: list[dict], rng: random.Random) -> dict[str, list]:
    """ファイル単位でまとめ直し、型チェックOKの単位だけを含むファイル内容に穴を空ける。"""
    out = {"train": [], "val": []}
    by_file: dict[tuple, list] = {}
    for u in units:
        by_file.setdefault((u["pack"], u["file"]), []).append(u)
    for (pack, file), us in by_file.items():
        good = [u for u in us if u["tsc_ok"]]
        if not good:
            continue
        text = (good[0]["imports"] + "\n\n" if good[0]["imports"] else "") + "\n\n".join(u["code"].rstrip() for u in good) + "\n"
        n = max(1, min(12, len(text.splitlines()) // 25))
        split = "val" if is_val(file) else "train"
        for _ in range(n):
            lines = text.splitlines(keepends=True)
            i = rng.randrange(len(lines))
            kind = rng.choice(["line_rest", "line", "lines"])
            start = sum(len(l) for l in lines[:i])
            if kind == "line_rest":
                body = lines[i].rstrip("\n")
                start += rng.randint(len(body) - len(body.lstrip()), max(len(body) - len(body.lstrip()), len(body) - 1))
                end = sum(len(l) for l in lines[:i]) + len(body)
            elif kind == "line":
                end = start + len(lines[i].rstrip("\n"))
                start += len(lines[i]) - len(lines[i].lstrip())
            else:
                j = min(len(lines), i + rng.randint(2, 8))
                end = sum(len(l) for l in lines[:j]) - 1
            if end <= start:
                continue
            path = f"{pack}/{file}" if rng.random() < 0.3 else None
            out[split].append({"prompt": fim_prompt(text[:start][-4000:], text[end:][:1500], path),
                               "completion": text[start:end] + EOT, "kind": f"script_fim_{kind}"})
    return out


# 正しいコードを壊す典型パターン（実際にユーザーのコードや qwen3 の出力で見られた誤り）
MUTATIONS = [
    ("old_startup", r"system\.beforeEvents\.startup", "world.beforeEvents.worldInitialize"),
    ("old_worldload", r"world\.afterEvents\.worldLoad", "world.afterEvents.worldInitialize"),
    ("old_runcommand", r"\.runCommand\(", ".runCommandAsync("),
    ("enum_as_string", r"CustomCommandParamType\.(\w+)", r'"\1"'),
    ("enum_as_string", r"EquipmentSlot\.(\w+)", lambda m: f'EquipmentSlot.{m.group(1).lower()}'),
    ("no_player_guard", r"if \(!\(\w+ instanceof Player\)\) return;?\n", ""),
    ("ui_wrong_module", r'from "@minecraft/server-ui"', 'from "@minecraft/server"'),
    ("typo_method", r"\.sendMessage\(", ".sendMesage("),
    ("typo_method", r"\.getComponent\(", ".getComponents("),
    ("typo_event", r"afterEvents\.entityHurt", "afterEvents.entityHurted"),
    ("typo_event", r"afterEvents\.playerSpawn", "afterEvents.playerJoin"),
]

REPAIR_TEMPLATE = """次のスクリプトを型チェック（@minecraft/server 2.10.0）したところ、エラーが出ました。修正したファイル全体を出力してください。

{code}

エラー:
{errors}
{hints}"""


def build_script_repair(units: list[dict], rng: random.Random, per_unit: int = 2) -> dict[str, list]:
    out = {"train": [], "val": []}
    kinds = Counter()
    for u in units:
        if not u["tsc_ok"] or u["file"].endswith(".ts"):
            continue
        original = (u["imports"] + "\n\n" if u["imports"] else "") + u["code"]
        cands = [m for m in MUTATIONS if re.search(m[1], original)]
        rng.shuffle(cands)
        for name, pat, repl in cands[:per_unit]:
            broken = re.sub(pat, repl, original, count=1)
            if broken == original:
                continue
            errors = [str(i) for i in check_scripts({u["file"]: broken})]
            if not errors:
                continue  # 型チェックで検出できない改変は教材にしない
            hints = repair_hints(errors)
            user = REPAIR_TEMPLATE.format(
                code=f"### {u['file']}\n```js\n{broken.rstrip()}\n```",
                errors="\n".join(f"- {e}" for e in errors[:8]),
                hints=f"\nヒント:\n{hints}" if hints and rng.random() < 0.5 else "")
            split = "val" if is_val(u["file"]) else "train"
            out[split].append({"prompt": chat_prompt(user), "kind": f"script_repair_{name}",
                               "completion": chat_completion(f"### {u['file']}\n```js\n{original.rstrip()}\n```")})
            kinds[name] += 1
    print("  script_repair:", dict(kinds))
    return out


MIGRATED = PROJECT_ROOT / "data" / "user" / "migrated.jsonl"
MIGRATE_TEMPLATE = "このスクリプトを @minecraft/server 2.x の安定版で動くように直して。\n\n### {file}\n```js\n{code}\n```"


def build_migration(rng: random.Random) -> dict[str, list]:
    """旧API（1.x / beta）のユーザーコード → qwen3 が移行し型チェックを通ったもの。エラー付き/なしを混ぜる。"""
    out = {"train": [], "val": []}
    for line in MIGRATED.read_text(encoding="utf-8").splitlines() if MIGRATED.exists() else []:
        m = json.loads(line)
        user = MIGRATE_TEMPLATE.format(file=m["file"], code=m["before"].rstrip())
        if m["before_errors"] and rng.random() < 0.5:
            user += "\n\n型チェックのエラー:\n" + "\n".join(f"- {e}" for e in m["before_errors"][:8])
        split = "val" if is_val(m["file"]) else "train"
        out[split].append({"prompt": chat_prompt(user), "kind": "script_migrate",
                           "completion": chat_completion(f"### {m['file']}\n```js\n{m['after'].rstrip()}\n```")})
    return out


def build_user_json_fim(rng: random.Random) -> dict[str, list]:
    out = {"train": [], "val": []}
    validator = SchemaValidator()
    for pack in PACKS:
        for sub, n in (("entities", 4), ("items", 2), ("blocks", 3), ("spawn_rules", 2), ("loot_tables", 1)):
            for p in sorted((pack / sub).rglob("*.json")) if (pack / sub).exists() else []:
                if check_file(p, validator).errors:
                    continue  # 構文エラー等のあるファイルはお手本にしない
                rel = f"behavior_pack/{p.relative_to(pack).as_posix()}"
                split = "val" if is_val(f"{pack.name}/{rel}") else "train"
                out[split] += [dict(s, kind=f"user_json_{s['kind']}") for s in
                               json_fim_samples(p.read_text(encoding="utf-8", errors="replace"), rel, rng, n)]
    return out


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(3)
    units_list = [json.loads(l) for l in UNITS.read_text(encoding="utf-8").splitlines() if l]
    units = {unit_id(u): u for u in units_list}
    parts = {
        "script_fim": build_script_fim(units_list, rng),
        "script_repair": build_script_repair(units_list, rng),
        "user_json_fim": build_user_json_fim(rng),
    }
    if BACKTRANS.exists():
        parts["script_gen"] = build_script_gen(units, rng)
    if MIGRATED.exists():
        parts["script_migrate"] = build_migration(rng)
    for name, splits in parts.items():
        for split, rows in splits.items():
            rng.shuffle(rows)
            path = SFT / f"v3_{name}_{split}.jsonl"
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{path.name}: {len(rows)}")


if __name__ == "__main__":
    main()
