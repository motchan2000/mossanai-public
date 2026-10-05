"""公式スキーマで拾えない「意味」のチェック。LLM出力で起きやすいミスを狙う。

1. 存在しないコンポーネント名（minecraft:helth 等）→ エラー + 候補提示
2. イベントが参照する component_group が存在しない → エラー
3. スキーマが "Dynamic value"（型定義なし）のコンポーネントは、vanillaでの実際の使われ方から
   フィールド名と型のプロファイルを作り、外れたら警告
"""
from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from bedrock_tools.jsonc import parse_lenient
from bedrock_tools.schema_validator import PROJECT_ROOT, default_schema_root

IDENTIFIER_RE = re.compile(r"^[a-z0-9_.\-]+:[a-z0-9_.\-/]+$")


@dataclass
class SemanticIssue:
    severity: str  # "error" | "warning"
    path: str
    message: str
    # 旧形式の format_version でもエラーのままにする（どのバージョンでもゲームが読み込めない誤り）
    strict: bool = False

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


# ---------------------------------------------------------------------------
# 既知コンポーネント名（公式スキーマの properties から）
# ---------------------------------------------------------------------------
COMPONENT_SCHEMAS = {
    "entity": ("server/entity", "Entity component definitions.json"),
    "item": ("server/item", "Item Components.json"),
    "block": ("server/block", "Components.json"),
}


def _is_dynamic(spec: dict) -> bool:
    return spec.get("description") == "Dynamic value" and not (set(spec) - {"description", "x-ordinal-index"})


@lru_cache(maxsize=None)
def component_catalog(kind: str) -> tuple[frozenset[str], frozenset[str]]:
    """(既知のコンポーネント名すべて, そのうち型定義がDynamicなもの)。全バージョンの和集合。"""
    root = default_schema_root()
    sub, fname = COMPONENT_SCHEMAS[kind]
    names, dynamic_latest = set(), set()
    files = sorted((root / sub).glob(f"*/{fname}"), key=lambda p: p.parent.name)
    for p in files:
        props = json.loads(p.read_text(encoding="utf-8")).get("properties", {})
        names |= set(props)
    if files:
        stable = [p for p in files if p.parent.name[0].isdigit()] or files
        latest = max(stable, key=lambda p: tuple(int(x) for x in re.findall(r"\d+", p.parent.name)))
        props = json.loads(latest.read_text(encoding="utf-8")).get("properties", {})
        dynamic_latest = {k for k, v in props.items() if _is_dynamic(v)}
    if kind == "block":  # ブロックの permutations 等で使われる追加名（Traits.json）
        for p in (root / sub).glob("*/Traits.json"):
            names |= set(json.loads(p.read_text(encoding="utf-8")).get("properties", {}))
    return frozenset(names), frozenset(dynamic_latest)


# ---------------------------------------------------------------------------
# vanillaから推定するフィールドプロファイル（Dynamicコンポーネント用）
# ---------------------------------------------------------------------------
def _jtype(v) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "null"


@lru_cache(maxsize=1)
def vanilla_entity_profiles() -> dict[str, dict[str, frozenset[str]]]:
    """{component名: {フィールド名: 観測された型の集合}}（トップレベルフィールドのみ）"""
    roots = sorted((PROJECT_ROOT / "data" / "bedrock" / "raw").glob("bedrock-samples-*/behavior_pack/entities"))
    prof: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    if not roots:
        return {}
    for p in roots[-1].glob("*.json"):
        doc, err = parse_lenient(p.read_text(encoding="utf-8"))
        if err:
            continue
        ent = doc.get("minecraft:entity", {})
        blocks = [ent.get("components") or {}] + list((ent.get("component_groups") or {}).values())
        for comps in blocks:
            for name, body in comps.items():
                if isinstance(body, dict):
                    prof[name]  # 空でも「観測済み」として登録
                    for field, v in body.items():
                        prof[name][field].add(_jtype(v))
    return {k: {f: frozenset(t) for f, t in v.items()} for k, v in prof.items()}


# ---------------------------------------------------------------------------
# チェック本体
# ---------------------------------------------------------------------------
def _suggest(name: str, known) -> str:
    cand = difflib.get_close_matches(name, known, n=3, cutoff=0.75)
    return f"（候補: {', '.join(cand)}）" if cand else ""


def _check_components(comps: dict, kind: str, path: str, issues: list[SemanticIssue]) -> None:
    known, dynamic = component_catalog(kind)
    profiles = vanilla_entity_profiles() if kind == "entity" else {}
    for name, body in comps.items():
        p = f"{path}/{name}"
        if name == "minecraft:custom_components" and kind in ("item", "block"):
            if not (isinstance(body, list) and all(isinstance(x, str) for x in body)):
                issues.append(SemanticIssue("error", p, "文字列の配列（例: [\"ns:name\"]）で書きます。"
                                                        "オブジェクトを書くとゲームが読み込めません", strict=True))
                continue
            # 1.21.90 より前の方式（現在はカスタムコンポーネント名を components に直接書く）
            issues.append(SemanticIssue("warning", p, "旧方式です。format_version 1.21.90 以降はカスタムコンポーネント名"
                                                      "（\"ns:name\": {}）を components に直接書きます"))
            continue
        if name.startswith("minecraft:"):
            # スキーマの名前一覧は不完全（例: minecraft:pushable が無い）ため vanilla での使用実績も既知扱い
            if name not in known and name not in profiles:
                # ブロックは未知のコンポーネントが1つでもあると定義全体の読み込みに失敗する（BDS 1.26 で確認）。
                # アイテム・エンティティは無視されるだけなので、旧形式では警告に下げてよい
                issues.append(SemanticIssue("error", p, f"存在しないコンポーネントです{_suggest(name, known)}",
                                            strict=kind == "block"))
                continue
        elif kind == "entity":
            issues.append(SemanticIssue("error", p, "entity のコンポーネントは minecraft: で始まる必要があります"))
            continue
        # 型定義なしコンポーネント: vanillaプロファイルで補完チェック（警告）
        if name in dynamic and name in profiles and isinstance(body, dict):
            prof = profiles[name]
            for field, v in body.items():
                if field not in prof:
                    if prof:  # vanillaで1つもフィールドが無いものは判断しない
                        issues.append(SemanticIssue(
                            "warning", f"{p}/{field}",
                            f"vanillaでは使われていないフィールドです{_suggest(field, prof.keys())}"))
                elif _jtype(v) not in prof[field]:
                    issues.append(SemanticIssue(
                        "warning", f"{p}/{field}",
                        f"vanillaでは {'/'.join(sorted(prof[field]))} 型で使われています（現在: {_jtype(v)}）"))


def _event_refs(node, path: str):
    """イベント定義内の add/remove.component_groups を再帰的に集める（sequence/randomize対応）。"""
    if isinstance(node, dict):
        for op in ("add", "remove"):
            groups = node.get(op, {}).get("component_groups") if isinstance(node.get(op), dict) else None
            if isinstance(groups, list):
                for i, g in enumerate(groups):
                    yield f"{path}/{op}/component_groups/{i}", g
        for k, v in node.items():
            if k in ("sequence", "randomize", "first_valid") and isinstance(v, list):
                for i, child in enumerate(v):
                    yield from _event_refs(child, f"{path}/{k}/{i}")


def _check_identifier(desc: dict, path: str, issues: list[SemanticIssue]) -> None:
    ident = desc.get("identifier")
    if not isinstance(ident, str) or not IDENTIFIER_RE.match(ident):
        issues.append(SemanticIssue("error", f"{path}/identifier",
                                    f'identifier は "namespace:name"（小文字）の形式にしてください: {ident!r}'))
    elif ident.startswith("minecraft:"):
        issues.append(SemanticIssue("warning", f"{path}/identifier",
                                    "minecraft: 名前空間はバニラ用です。独自の名前空間を推奨します"))


def check_entity(doc: dict) -> list[SemanticIssue]:
    issues: list[SemanticIssue] = []
    ent = doc.get("minecraft:entity")
    if not isinstance(ent, dict):
        return issues
    base = "/minecraft:entity"
    _check_identifier(ent.get("description") or {}, f"{base}/description", issues)
    _check_components(ent.get("components") or {}, "entity", f"{base}/components", issues)
    groups = ent.get("component_groups") or {}
    for g, comps in groups.items():
        if isinstance(comps, dict):
            _check_components(comps, "entity", f"{base}/component_groups/{g}", issues)
    for ev, body in (ent.get("events") or {}).items():
        for p, g in _event_refs(body, f"{base}/events/{ev}"):
            if g not in groups:
                # ゲームは無視するだけで落ちない（vanillaにも残骸がある）ので警告。ただし意図した挙動は起きない。
                issues.append(SemanticIssue("warning", p, f"component_group {g!r} が定義されていないため、この操作は何も起こしません"
                                                          f"{_suggest(g, groups.keys())}"))
    return issues


def check_item_or_block(doc: dict) -> list[SemanticIssue]:
    issues: list[SemanticIssue] = []
    for key, kind in (("minecraft:item", "item"), ("minecraft:block", "block")):
        body = doc.get(key)
        if not isinstance(body, dict):
            continue
        _check_identifier(body.get("description") or {}, f"/{key}/description", issues)
        _check_components(body.get("components") or {}, kind, f"/{key}/components", issues)
        for i, perm in enumerate(body.get("permutations", []) if kind == "block" else []):
            _check_components(perm.get("components", {}), kind, f"/{key}/permutations/{i}/components", issues)
    return issues


@lru_cache(maxsize=1)
def spawn_rule_keys() -> tuple[frozenset[str], frozenset[str]]:
    """(minecraft:spawn_rules 直下のキー, conditions 内のキー)。公式スキーマとvanillaの使用実績の和集合。"""
    root = default_schema_root()
    top, cond = {"description", "conditions"}, set()
    for p in list(root.glob("server/spawn/*/Spawn Rules.json")) + list(root.glob("client_server/spawn/*/Spawn Rules.json")):
        top |= set(json.loads(p.read_text(encoding="utf-8")).get("properties", {}))
    for p in root.glob("server/spawn/*/Spawn BiomeConditions.json"):
        cond |= set(json.loads(p.read_text(encoding="utf-8")).get("properties", {}))
    vanilla = sorted((PROJECT_ROOT / "data" / "bedrock" / "raw").glob("bedrock-samples-*/behavior_pack/spawn_rules"))
    for p in (vanilla[-1].glob("*.json") if vanilla else []):
        doc, err = parse_lenient(p.read_text(encoding="utf-8"))
        for c in (doc or {}).get("minecraft:spawn_rules", {}).get("conditions", []) if not err else []:
            cond |= set(c)
    return frozenset(top), frozenset(cond)


def check_spawn_rules(doc: dict) -> list[SemanticIssue]:
    """公式スキーマは未知のキーを禁止していないので、存在しないキー（LLMの創作）をここで検出する。"""
    issues: list[SemanticIssue] = []
    body = doc.get("minecraft:spawn_rules")
    if not isinstance(body, dict):
        return issues
    top, cond = spawn_rule_keys()
    for k in body:
        if k not in top:
            issues.append(SemanticIssue("error", f"/minecraft:spawn_rules/{k}",
                                        f"spawn_rules に {k!r} というキーはありません{_suggest(k, top)}"))
    _check_identifier(body.get("description") or {}, "/minecraft:spawn_rules/description", issues)
    for i, c in enumerate(body.get("conditions") or []):
        for k in c if isinstance(c, dict) else []:
            if k not in cond:
                issues.append(SemanticIssue("error", f"/minecraft:spawn_rules/conditions/{i}/{k}",
                                            f"存在しない条件コンポーネントです{_suggest(k, cond)}"))
    return issues


def check_semantics(doc) -> list[SemanticIssue]:
    if not isinstance(doc, dict):
        return []
    if "minecraft:entity" in doc:
        return check_entity(doc)
    if "minecraft:item" in doc or "minecraft:block" in doc:
        return check_item_or_block(doc)
    if "minecraft:spawn_rules" in doc:
        return check_spawn_rules(doc)
    return []
