"""Mojang公式JSON Schema（bedrock-samples/metadata/json_schemas）によるアドオンJSONの検証。

- ファイル内容からドキュメント種別（entity/item/block/...）を判定
- format_version に対応するスキーマのバージョンを選択（それ以下で最新）
- スキーマ間の相対 $ref は referencing の Registry で解決
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

from jsonschema import Draft7Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

from bedrock_tools.jsonc import parse_lenient

BASE_URI = "https://bedrock.schemas"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def default_schema_root() -> Path:
    candidates = sorted((PROJECT_ROOT / "data" / "bedrock" / "raw").glob("bedrock-samples-*/metadata/json_schemas"))
    if not candidates:
        raise FileNotFoundError("json_schemas が見つかりません。data/bedrock/fetch_samples.py を実行してください")
    return candidates[-1]


@dataclass(frozen=True)
class DocType:
    name: str
    schema_dir: str  # json_schemas からの相対ディレクトリ（直下にバージョンディレクトリが並ぶ）
    schema_file: str
    key: str | None  # このキーの値をスキーマで検証。Noneならファイル全体
    legacy_before: str | None = None  # これより古い format_version は旧仕様（現行スキーマは参考扱い）


DOC_TYPES = [
    DocType("entity", "server/entity", "ActorDocument.json", "minecraft:entity"),
    DocType("item", "server/item", "ItemDocument.json", "minecraft:item", legacy_before="1.20.0"),
    DocType("block", "server/block", "Blocks.json", "minecraft:block"),
    DocType("spawn_rules", "server/spawn", "Spawn Rules.json", "minecraft:spawn_rules"),
    DocType("biome", "server/biome", "Biome Document.json", None),
    DocType("client_biome", "client/biome", "Client Biome Document.json", None),
    DocType("manifest", "client_server/packaging", "Manifest.json", None),
]


def detect_type(doc: dict) -> DocType | None:
    if not isinstance(doc, dict):
        return None
    if "header" in doc and "modules" in doc:
        # 公式スキーマは format_version 3 用。format 2 は manifest_checker で検証する
        if doc.get("format_version") == 3:
            return next(t for t in DOC_TYPES if t.name == "manifest")
        return None
    for t in DOC_TYPES:
        if t.key and t.key in doc:
            return t
    if "minecraft:biome" in doc:
        return next(t for t in DOC_TYPES if t.name == "biome")
    if "minecraft:client_biome" in doc:
        return next(t for t in DOC_TYPES if t.name == "client_biome")
    return None


def _version_key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v))


@dataclass
class ValidationIssue:
    path: str  # JSON Pointer風 (/minecraft:entity/components/minecraft:health)
    message: str

    def __str__(self) -> str:
        return f"{self.path or '/'}: {self.message}"


@dataclass
class ValidationResult:
    doc_type: str | None
    schema: str | None
    issues: list[ValidationIssue] = field(default_factory=list)
    parse_error: str | None = None
    legacy: bool = False  # 旧仕様のformat_version（issuesは参考情報）

    @property
    def ok(self) -> bool:
        return self.parse_error is None and not self.issues


class SchemaValidator:
    def __init__(self, schema_root: Path | None = None):
        self.root = Path(schema_root) if schema_root else default_schema_root()
        self.registry = Registry(retrieve=self._retrieve)

    @lru_cache(maxsize=None)
    def _load(self, rel: str) -> dict:
        data = _one_of_to_any_of(json.loads((self.root / rel).read_text(encoding="utf-8")))
        data["$id"] = f"{BASE_URI}/{rel}"
        return data

    def _retrieve(self, uri: str) -> Resource:
        rel = unquote(uri.removeprefix(BASE_URI + "/").split("#")[0])
        return Resource.from_contents(self._load(rel), default_specification=DRAFT7)

    def pick_schema(self, t: DocType, format_version: str | None) -> str:
        """format_version 以下で最新のスキーマバージョンを選ぶ。該当なしなら最古、betaは最終手段。"""
        dirs = [d for d in (self.root / t.schema_dir).iterdir() if (d / t.schema_file).exists()]
        stable = sorted((d for d in dirs if d.name[0].isdigit()), key=lambda d: _version_key(d.name))
        if not stable:
            chosen = dirs[0]
        elif format_version is None or str(format_version)[0:1].isdigit() is False:
            chosen = stable[-1]
        else:
            fv = _version_key(str(format_version))
            older = [d for d in stable if _version_key(d.name) <= fv]
            chosen = older[-1] if older else stable[0]
        return f"{t.schema_dir}/{chosen.name}/{t.schema_file}"

    def validate(self, doc, doc_type: str | None = None) -> ValidationResult:
        t = next(x for x in DOC_TYPES if x.name == doc_type) if doc_type else detect_type(doc)
        if t is None:
            return ValidationResult(None, None)  # 対応スキーマなし（検証対象外）
        fv = doc.get("format_version") if isinstance(doc, dict) else None
        rel = self.pick_schema(t, None if t.name == "manifest" else fv)
        schema = self._load(rel)
        target = doc[t.key] if t.key else doc
        validator = Draft7Validator(schema, registry=self.registry)
        issues = []
        for err in sorted(validator.iter_errors(target), key=lambda e: list(e.absolute_path)):
            prefix = [t.key] if t.key else []
            path = "/" + "/".join(str(p) for p in prefix + list(err.absolute_path))
            issues.append(ValidationIssue(path, _short_message(err)))
        legacy = False
        if isinstance(fv, str) and fv[:1].isdigit():
            # 公式スキーマの最古版より古い format_version は、新しい版のスキーマで判定することになるため参考扱い
            # （例: ブロックのスキーマは 1.26.20 しかなく、1.21.x の数値ステート等を誤ってエラーにする）
            oldest = self.oldest_version(t)
            legacy = bool((t.legacy_before and _version_key(fv) < _version_key(t.legacy_before))
                          or (oldest and _version_key(fv) < _version_key(oldest)))
        return ValidationResult(t.name, rel, issues, legacy=legacy)

    @lru_cache(maxsize=None)
    def oldest_version(self, t: DocType) -> str | None:
        dirs = [d.name for d in (self.root / t.schema_dir).iterdir()
                if (d / t.schema_file).exists() and d.name[0].isdigit()]
        return min(dirs, key=_version_key) if dirs else None

    def validate_text(self, text: str, doc_type: str | None = None) -> ValidationResult:
        doc, err = parse_lenient(text)
        if err:
            return ValidationResult(None, None, parse_error=err)
        return self.validate(doc, doc_type)


def _one_of_to_any_of(node):
    """公式スキーマはエディタ補完用に書かれており、oneOf の選択肢が重なっている
    （例: フィルタの単体形式と配列形式の両方に一致する）ため、anyOf として扱う。"""
    if isinstance(node, dict):
        return {("anyOf" if k == "oneOf" else k): _one_of_to_any_of(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_one_of_to_any_of(v) for v in node]
    return node


def _short_message(err) -> str:
    """oneOf/anyOf の長大なメッセージを、最も具体的な子エラーに要約する。"""
    if err.context:
        best = max(err.context, key=lambda e: len(e.absolute_path))
        return f"{best.message[:200]}"
    return err.message[:300]
