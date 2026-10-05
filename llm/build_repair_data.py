"""「検証エラー → 修正」指示データの作成（CLIの自己修正ループ用）。

正しいvanillaファイル（trainに分けたもの）に意図的な誤りを1つ入れ、
bedrock_tools の検証で実際にエラー/警告が出たものだけを採用する。
入力: ファイル + 検証メッセージ / 出力: 修正済みのファイル全体

出力: data/sft/repair_train.jsonl（--split val で repair_val.jsonl）
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

from bedrock_tools.check import check_file
from bedrock_tools.jsonc import dump, parse_lenient
from bedrock_tools.manifest import new_pack_pair
from bedrock_tools.schema_validator import SchemaValidator
from llm.chat import chat_completion, chat_prompt, code_block
from llm.fim import PROJECT_ROOT
from llm.fim_tasks import is_val

SOURCES = ["behavior_pack/entities/", "behavior_pack/items/", "behavior_pack/blocks/", "behavior_pack/spawn_rules/"]


# ---------------------------------------------------------------------------
# 誤りの入れ方（それぞれ (壊したテキスト, 種類) を返す。適用できなければNone）
# ---------------------------------------------------------------------------
def _components(doc: dict) -> list[dict]:
    for key in ("minecraft:entity", "minecraft:item", "minecraft:block"):
        body = doc.get(key)
        if isinstance(body, dict):
            out = [body.get("components") or {}]
            out += [g for g in (body.get("component_groups") or {}).values() if isinstance(g, dict)]
            return [c for c in out if c]
    return []


def typo(name: str, rng: random.Random) -> str:
    head, tail = name.split(":", 1)
    i = rng.randrange(len(tail))
    op = rng.choice(["drop", "swap", "dup"])
    if op == "drop" and len(tail) > 3:
        tail = tail[:i] + tail[i + 1:]
    elif op == "swap" and i < len(tail) - 1:
        tail = tail[:i] + tail[i + 1] + tail[i] + tail[i + 2:]
    else:
        tail = tail[:i] + tail[i] + tail[i:]
    return f"{head}:{tail}"


def mut_component_typo(doc, rng):
    comps = [c for c in _components(doc) if any(k.startswith("minecraft:") for k in c)]
    if not comps:
        return None
    doc = copy.deepcopy(doc)
    comps = [c for c in _components(doc) if any(k.startswith("minecraft:") for k in c)]
    c = rng.choice(comps)
    old = rng.choice([k for k in c if k.startswith("minecraft:")])
    new = typo(old, rng)
    if new == old:
        return None
    items = [(new if k == old else k, v) for k, v in c.items()]
    c.clear()
    c.update(items)
    return dump(doc), "component_typo"


def mut_wrong_type(doc, rng):
    """数値を文字列に、真偽値を文字列に変える。"""
    doc = copy.deepcopy(doc)
    targets = []
    for c in _components(doc):
        for name, body in c.items():
            if isinstance(body, dict):
                for k, v in body.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) or isinstance(v, bool):
                        targets.append((body, k))
    if not targets:
        return None
    body, k = rng.choice(targets)
    v = body[k]
    body[k] = str(v).lower() if isinstance(v, bool) else str(v)
    return dump(doc), "wrong_type"


def mut_syntax(text, rng):
    lines = text.split("\n")
    # 末尾カンマはBedrockでは許容されるので対象外
    kind = rng.choice(["drop_comma", "missing_colon", "drop_brace", "unquote_key"])
    cands = {
        "drop_comma": [i for i, l in enumerate(lines) if l.rstrip().endswith(",")],
        "missing_colon": [i for i, l in enumerate(lines) if re.match(r'^\s*"[^"]+"\s*:\s*\S', l)],
        "drop_brace": [i for i, l in enumerate(lines) if l.strip() in ("}", "},") and i < len(lines) - 1],
        "unquote_key": [i for i, l in enumerate(lines) if re.match(r'^\s*"[a-z_]+"\s*:', l)],
    }[kind]
    if not cands:
        return None
    i = rng.choice(cands)
    if kind == "drop_comma":
        lines[i] = lines[i].rstrip()[:-1]
    elif kind == "missing_colon":
        m = re.match(r'^(\s*"[^"]+")\s*:\s*(.*)$', lines[i])
        lines[i] = f"{m.group(1)} {m.group(2)}"
    elif kind == "drop_brace":
        del lines[i]
    else:
        lines[i] = re.sub(r'^(\s*)"([a-z_]+)"', r"\1\2", lines[i], count=1)
    return "\n".join(lines), f"syntax_{kind}"


def mut_event_group(doc, rng):
    ent = doc.get("minecraft:entity")
    if not isinstance(ent, dict):
        return None
    doc = copy.deepcopy(doc)
    ent = doc["minecraft:entity"]
    refs = []
    for ev in (ent.get("events") or {}).values():
        stack = [ev]
        while stack:
            node = stack.pop()
            if not isinstance(node, dict):
                continue
            for op in ("add", "remove"):
                gl = (node.get(op) or {}).get("component_groups") if isinstance(node.get(op), dict) else None
                if isinstance(gl, list) and gl:
                    refs.append(gl)
            for k in ("sequence", "randomize", "first_valid"):
                if isinstance(node.get(k), list):
                    stack += node[k]
    refs = [gl for gl in refs if any(g in (ent.get("component_groups") or {}) for g in gl)]
    if not refs:
        return None
    gl = rng.choice(refs)
    i = rng.choice([i for i, g in enumerate(gl) if g in ent["component_groups"]])
    new = typo(gl[i] if ":" in gl[i] else f"x:{gl[i]}", rng).removeprefix("x:")
    if new == gl[i]:
        return None
    gl[i] = new
    return dump(doc), "event_group_typo"


def manifest_samples(rng: random.Random, n: int) -> list[tuple[str, str, str]]:
    """(壊したmanifest, 正しいmanifest, 種類)。正解は壊す前の生成結果そのもの。"""
    out = []
    for _ in range(n):
        bp, _ = new_pack_pair(f"Pack{rng.randint(1, 999)}", "", script_entry="scripts/main.js" if rng.random() < .6 else None)
        good = dump(bp)
        bad = copy.deepcopy(bp)
        kind = rng.choice(["dup_uuid", "bad_version", "bad_module_version", "bad_entry"])
        if kind == "dup_uuid":
            bad["modules"][0]["uuid"] = bad["header"]["uuid"]
        elif kind == "bad_version":
            bad["header"]["version"] = rng.choice(["1.0.0", [1, 0], "v1"])
        elif kind == "bad_module_version" and any("module_name" in d for d in bad["dependencies"]):
            next(d for d in bad["dependencies"] if "module_name" in d)["version"] = rng.choice(["3.0.0", "1.99.0", "2.10"])
        elif kind == "bad_entry" and len(bad["modules"]) > 1:
            bad["modules"][1]["entry"] = rng.choice(["main.js", "scripts/main.ts", "index.js"])
        else:
            bad["modules"][0]["uuid"] = bad["header"]["uuid"]
            kind = "dup_uuid"
        # 正解側は重複UUIDのみ新規UUIDに差し替わる（元のUUIDを答えとして使う）
        out.append((dump(bad), good, f"manifest_{kind}"))
    return out


USER_TEMPLATE = """次のファイルを検証したところ、以下の問題が見つかりました。修正したファイル全体を出力してください。

ファイル: {path}
{code}

検証結果:
{issues}"""


def issues_text(text: str, tmp: Path, validator: SchemaValidator) -> tuple[list[str], list[str]]:
    tmp.write_text(text, encoding="utf-8")
    rep = check_file(tmp, validator)
    return rep.errors, rep.warnings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--per_file", type=int, default=3)
    ap.add_argument("--max_chars", type=int, default=5000, help="これより長いファイルは使わない（context長対策）")
    ap.add_argument("--manifests", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()

    rng = random.Random(a.seed + (a.split == "val"))
    src = sorted((PROJECT_ROOT / "data" / "bedrock" / "raw").glob("bedrock-samples-*"))[-1]
    validator = SchemaValidator()
    tmp_dir = PROJECT_ROOT / "data" / "sft" / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / "x.json"
    mutators = [mut_component_typo, mut_wrong_type, mut_event_group]

    rows, kinds = [], Counter()
    for d in SOURCES:
        for p in sorted(src.glob(d + "**/*.json")):
            rel = p.relative_to(src).as_posix()
            if is_val(rel) != (a.split == "val"):
                continue
            doc, err = parse_lenient(p.read_text(encoding="utf-8"))
            if err:
                continue
            good = dump(doc)
            if len(good) > a.max_chars:
                continue
            base_errs, base_warns = issues_text(good, tmp, validator)
            if base_errs:  # 元から問題があるもの（旧形式等）は正解として使わない
                continue
            for _ in range(a.per_file):
                if rng.random() < 0.3:
                    res = mut_syntax(good, rng)
                else:
                    res = rng.choice(mutators)(doc, rng)
                if not res:
                    continue
                bad, kind = res
                errs, warns = issues_text(bad, tmp, validator)
                new_warns = [w for w in warns if w not in base_warns]
                problems = errs + new_warns
                if not problems:
                    continue  # 検証ツールが検出できない改変は教師にしない
                user = USER_TEMPLATE.format(path=f"{p.parent.parent.name}/{p.parent.name}/{p.name}",
                                            code=code_block(bad), issues="\n".join(f"- {x}" for x in problems[:8]))
                rows.append({"prompt": chat_prompt(user), "completion": chat_completion(code_block(good)),
                             "kind": kind, "path": rel})
                kinds[kind] += 1

    for bad, good, kind in manifest_samples(rng, a.manifests if a.split == "train" else a.manifests // 10):
        tmp_m = tmp_dir / "manifest.json"
        tmp_m.write_text(bad, encoding="utf-8")
        errs = check_file(tmp_m, validator).errors
        if not errs:
            continue
        user = USER_TEMPLATE.format(path="BP/manifest.json", code=code_block(bad),
                                    issues="\n".join(f"- {x}" for x in errs[:8]))
        rows.append({"prompt": chat_prompt(user), "completion": chat_completion(code_block(good)),
                     "kind": kind, "path": "manifest.json"})
        kinds[kind] += 1

    rng.shuffle(rows)
    out = PROJECT_ROOT / "data" / "sft" / f"repair_{a.split}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for f in tmp_dir.iterdir():
        f.unlink()
    tmp_dir.rmdir()
    print(out, len(rows), dict(kinds))


if __name__ == "__main__":
    main()
