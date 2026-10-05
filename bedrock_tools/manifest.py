"""manifest.json の生成と検証。

UUIDは必ずランダムで一意でなければならないので、LLMには書かせずここで生成する。
検証は format_version 2（バージョンが配列）と 3（文字列）の両方に対応。
"""
from __future__ import annotations

import re
import uuid as uuidlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from bedrock_tools.schema_validator import PROJECT_ROOT

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
MODULE_TYPES = {"data", "resources", "script", "client_data", "interface", "world_template", "skin_pack"}
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(-[0-9A-Za-z.-]+)?$")

# 新規パックの既定値（bedrock-samples v1.26.50.4 時点）
DEFAULT_MIN_ENGINE = [1, 21, 90]


@dataclass
class ManifestIssue:
    severity: str  # "error" | "warning"
    path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.path}: {self.message}"


@lru_cache(maxsize=1)
def script_module_versions() -> dict[str, list[str]]:
    """bedrock-samples の script_modules から、公開モジュール名ごとの実在バージョン一覧を作る。"""
    roots = sorted((PROJECT_ROOT / "data" / "bedrock" / "raw").glob("bedrock-samples-*/metadata/script_modules/@minecraft"))
    out: dict[str, list[str]] = {}
    if not roots:
        return out
    for p in roots[-1].glob("*.json"):
        base, ver = p.stem.split("_", 1)
        if base.endswith("-bindings"):
            continue
        out.setdefault(f"@minecraft/{base}", []).append(ver)
    return out


def latest_stable(module_name: str) -> str | None:
    vers = [v for v in script_module_versions().get(module_name, []) if "-" not in v]
    return max(vers, key=lambda v: tuple(map(int, v.split(".")))) if vers else None


def _version_ok(v, fmt: int) -> bool:
    if isinstance(v, list):
        return len(v) == 3 and all(isinstance(x, int) and x >= 0 for x in v)
    if isinstance(v, str):
        return fmt >= 3 and bool(SEMVER_RE.match(v))
    return False


def check_manifest(doc: dict) -> list[ManifestIssue]:
    issues: list[ManifestIssue] = []

    def err(path, msg):
        issues.append(ManifestIssue("error", path, msg))

    def warn(path, msg):
        issues.append(ManifestIssue("warning", path, msg))

    fmt = doc.get("format_version")
    if fmt not in (1, 2, 3):
        err("/format_version", f"format_version は 2 か 3 である必要があります（現在: {fmt!r}）")
        fmt = 2
    elif fmt == 1:
        warn("/format_version", "format_version 1 は旧形式です。2 以上を推奨します")

    header = doc.get("header")
    if not isinstance(header, dict):
        err("/header", "header がありません")
        header = {}
    for key in ("name", "description", "uuid", "version"):
        if key not in header:
            (err if key in ("uuid", "version", "name") else warn)(f"/header/{key}", f"header.{key} がありません")
    if "version" in header and not _version_ok(header["version"], fmt):
        err("/header/version", f"バージョン形式が不正です: {header['version']!r}")

    seen: dict[str, str] = {}

    def check_uuid(value, path):
        if not isinstance(value, str) or not UUID_RE.match(value):
            err(path, f"UUIDの形式が不正です: {value!r}")
            return
        low = value.lower()
        if low in seen:
            err(path, f"UUIDが {seen[low]} と重複しています")
        else:
            seen[low] = path

    if "uuid" in header:
        check_uuid(header["uuid"], "/header/uuid")

    modules = doc.get("modules")
    if not isinstance(modules, list) or not modules:
        err("/modules", "modules が空です")
        modules = []
    types = set()
    for i, m in enumerate(modules):
        p = f"/modules/{i}"
        t = m.get("type")
        types.add(t)
        if t not in MODULE_TYPES:
            err(f"{p}/type", f"未知のモジュール種別です: {t!r}（{', '.join(sorted(MODULE_TYPES))}）")
        check_uuid(m.get("uuid"), f"{p}/uuid")
        if "version" not in m or not _version_ok(m["version"], fmt):
            err(f"{p}/version", f"バージョン形式が不正です: {m.get('version')!r}")
        if t == "script":
            if m.get("language") != "javascript":
                err(f"{p}/language", 'script モジュールには "language": "javascript" が必要です')
            entry = m.get("entry")
            if not isinstance(entry, str) or not entry.startswith("scripts/") or not entry.endswith(".js"):
                err(f"{p}/entry", f'entry は "scripts/xxx.js" の形式にしてください（現在: {entry!r}）')

    if {"data", "resources"} <= types:
        warn("/modules", "1つのパックに data と resources が混在しています。BP/RP は分けるのが一般的です")
    if types & {"data", "resources"} and "min_engine_version" not in header:
        warn("/header/min_engine_version", "min_engine_version の指定を推奨します")

    dep_modules = set()
    for i, d in enumerate(doc.get("dependencies", [])):
        p = f"/dependencies/{i}"
        if "module_name" in d:
            name, ver = d["module_name"], d.get("version")
            dep_modules.add(name)
            known = script_module_versions().get(name)
            if known is not None and ver not in known:
                stable = latest_stable(name)
                err(f"{p}/version", f"{name} にバージョン {ver!r} は存在しません（最新安定版: {stable}）")
        elif "uuid" in d:
            if not isinstance(d["uuid"], str) or not UUID_RE.match(d["uuid"]):
                err(f"{p}/uuid", f"UUIDの形式が不正です: {d['uuid']!r}")
            elif d["uuid"].lower() in seen:
                err(f"{p}/uuid", "自分自身のUUIDに依存しています")
            if not _version_ok(d.get("version"), fmt):
                err(f"{p}/version", f"バージョン形式が不正です: {d.get('version')!r}")
        else:
            err(p, "dependency には uuid か module_name が必要です")

    if "script" in types and "@minecraft/server" not in dep_modules:
        warn("/dependencies", "script モジュールがありますが @minecraft/server への依存がありません")
    return issues


def check_pack_pair(bp: dict, rp: dict) -> list[ManifestIssue]:
    """BPとRPが互いを依存関係に持っているか（片方だけだとRPが自動適用されない）。"""
    issues = []
    bp_uuid, rp_uuid = bp.get("header", {}).get("uuid"), rp.get("header", {}).get("uuid")
    bp_deps = {d.get("uuid") for d in bp.get("dependencies", [])}
    rp_deps = {d.get("uuid") for d in rp.get("dependencies", [])}
    if rp_uuid not in bp_deps:
        issues.append(ManifestIssue("warning", "BP/dependencies", "BPがRPのheader.uuidに依存していません"))
    if bp_uuid not in rp_deps:
        issues.append(ManifestIssue("warning", "RP/dependencies", "RPがBPのheader.uuidに依存していません"))
    return issues


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def new_uuid() -> str:
    return str(uuidlib.uuid4())


def new_pack_pair(
    name: str,
    description: str = "",
    version: list[int] | None = None,
    min_engine_version: list[int] | None = None,
    script_entry: str | None = None,
    script_modules: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    """互いに依存し合うBP/RPのmanifestを生成する。script_entryを指定するとscriptモジュールを追加。"""
    version = version or [1, 0, 0]
    mev = min_engine_version or DEFAULT_MIN_ENGINE
    bp_uuid, rp_uuid = new_uuid(), new_uuid()

    bp = {
        "format_version": 2,
        "header": {"name": f"{name} BP", "description": description, "uuid": bp_uuid,
                   "version": version, "min_engine_version": mev},
        "modules": [{"type": "data", "uuid": new_uuid(), "version": version}],
        "dependencies": [{"uuid": rp_uuid, "version": version}],
    }
    if script_entry:
        bp["modules"].append({"type": "script", "language": "javascript", "uuid": new_uuid(),
                              "version": version, "entry": script_entry})
        mods = script_modules or {"@minecraft/server": latest_stable("@minecraft/server") or "2.0.0"}
        bp["dependencies"] += [{"module_name": m, "version": v} for m, v in mods.items()]
    rp = {
        "format_version": 2,
        "header": {"name": f"{name} RP", "description": description, "uuid": rp_uuid,
                   "version": version, "min_engine_version": mev},
        "modules": [{"type": "resources", "uuid": new_uuid(), "version": version}],
        "dependencies": [{"uuid": bp_uuid, "version": version}],
    }
    return bp, rp


def scaffold(out_dir: Path, name: str, description: str = "", with_script: bool = False) -> tuple[Path, Path]:
    """BP/RPのディレクトリ構成とmanifestを作る。"""
    from bedrock_tools.jsonc import dump

    slug = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_") or "addon"
    bp_dir, rp_dir = out_dir / f"{slug}_BP", out_dir / f"{slug}_RP"
    bp, rp = new_pack_pair(name, description, script_entry="scripts/main.js" if with_script else None)
    for d in (bp_dir, rp_dir):
        d.mkdir(parents=True, exist_ok=False)
    (bp_dir / "manifest.json").write_text(dump(bp) + "\n", encoding="utf-8")
    (rp_dir / "manifest.json").write_text(dump(rp) + "\n", encoding="utf-8")
    for sub in ("entities", "items", "blocks", "loot_tables", "recipes", "spawn_rules"):
        (bp_dir / sub).mkdir()
    for sub in ("entity", "models/entity", "animations", "animation_controllers", "render_controllers",
                "textures/entity", "texts"):
        (rp_dir / sub).mkdir(parents=True)
    (rp_dir / "texts" / "en_US.lang").write_text(f"pack.name={name}\npack.description={description}\n", encoding="utf-8")
    (rp_dir / "texts" / "ja_JP.lang").write_text(f"pack.name={name}\npack.description={description}\n", encoding="utf-8")
    (rp_dir / "texts" / "languages.json").write_text('[\n  "en_US",\n  "ja_JP"\n]\n', encoding="utf-8")
    if with_script:
        (bp_dir / "scripts").mkdir()
        (bp_dir / "scripts" / "main.js").write_text(
            'import { world } from "@minecraft/server";\n\n'
            "world.afterEvents.worldLoad.subscribe(() => {\n"
            f'  world.sendMessage("{name} loaded");\n'
            "});\n",
            encoding="utf-8",
        )
    return bp_dir, rp_dir
