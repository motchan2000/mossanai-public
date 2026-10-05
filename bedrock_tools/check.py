"""パック（フォルダ）全体の検証: JSON構文・公式スキーマ・manifest・BP/RPペア。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bedrock_tools.jsonc import parse_lenient
from bedrock_tools.manifest import check_manifest, check_pack_pair
from bedrock_tools.schema_validator import SchemaValidator
from bedrock_tools.semantic import check_semantics


@dataclass
class FileReport:
    path: Path
    doc_type: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PackReport:
    files: list[FileReport] = field(default_factory=list)
    pair_warnings: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(len(f.errors) for f in self.files)

    @property
    def warning_count(self) -> int:
        return sum(len(f.warnings) for f in self.files) + len(self.pair_warnings)


def check_file(path: Path, validator: SchemaValidator) -> FileReport:
    rep = FileReport(path)
    doc, err = parse_lenient(path.read_text(encoding="utf-8", errors="replace"))
    if err:
        rep.errors.append(err)
        return rep
    if path.name == "manifest.json":
        rep.doc_type = "manifest"
        for i in check_manifest(doc):
            (rep.errors if i.severity == "error" else rep.warnings).append(f"{i.path}: {i.message}")
        if doc.get("format_version") != 3:
            return rep
    res = validator.validate(doc)
    rep.doc_type = rep.doc_type or res.doc_type
    target = rep.warnings if res.legacy else rep.errors
    if res.legacy and res.issues:
        rep.warnings.append(f"旧形式の format_version {doc.get('format_version')!r} です（現行スキーマでの指摘は参考）")
    target += [str(i) for i in res.issues]
    for i in check_semantics(doc):
        (rep.errors if i.severity == "error" and (i.strict or not res.legacy) else rep.warnings).append(str(i))
    return rep


def find_manifests(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("manifest.json") if ".git" not in p.parts)


def check_path(root: Path, validator: SchemaValidator | None = None) -> PackReport:
    validator = validator or SchemaValidator()
    report = PackReport()
    files = [root] if root.is_file() else sorted(
        p for p in root.rglob("*.json") if ".git" not in p.parts and "node_modules" not in p.parts)
    for p in files:
        report.files.append(check_file(p, validator))

    # BP/RPペア: data と resources の manifest が1つずつあれば相互依存を確認
    if root.is_dir():
        bp = rp = None
        for m in find_manifests(root):
            doc, err = parse_lenient(m.read_text(encoding="utf-8"))
            if err or not isinstance(doc, dict):
                continue
            types = {mod.get("type") for mod in doc.get("modules", [])}
            if "data" in types or "script" in types:
                bp = doc if bp is None else bp
            if "resources" in types:
                rp = doc if rp is None else rp
        if bp and rp:
            report.pair_warnings += [f"{i.path}: {i.message}" for i in check_pack_pair(bp, rp)]
    return report


def format_report(report: PackReport, root: Path, verbose: bool = False) -> str:
    lines = []
    for f in report.files:
        if not f.errors and not (verbose and f.warnings):
            continue
        rel = f.path.relative_to(root) if root.is_dir() else f.path.name
        lines.append(f"■ {rel}" + (f"  ({f.doc_type})" if f.doc_type else ""))
        lines += [f"  ✗ {e}" for e in f.errors]
        if verbose:
            lines += [f"  △ {w}" for w in f.warnings]
    if report.pair_warnings:
        lines.append("■ BP/RP の組み合わせ")
        lines += [f"  △ {w}" for w in report.pair_warnings]
    checked = sum(1 for f in report.files if f.doc_type)
    lines.append(f"\n{len(report.files)} ファイル（うちスキーマ検証 {checked}）: "
                 f"エラー {report.error_count} / 警告 {report.warning_count}")
    return "\n".join(lines)
