"""bedrock-samples/metadata/script_modules のAPI定義JSONを TypeScript宣言(.d.ts風)に変換する。

生のメタデータJSONは冗長（is_bind_type等）でトークン効率が悪く、しかもモデルに
学ばせたいのは「TypeScriptとしてのAPIの形」なので、宣言ファイルの形に整形して学習に使う。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PRIMITIVES = {
    "int8": "number", "int16": "number", "int32": "number", "int64": "number",
    "uint8": "number", "uint16": "number", "uint32": "number", "uint64": "number",
    "float": "number", "double": "number",
    "string": "string", "boolean": "boolean", "undefined": "undefined", "void": "void",
    "any": "any", "unknown": "unknown",
}


def ts_type(t: dict, ret: bool = False) -> str:
    name = t["name"]
    if name in PRIMITIVES:
        s = PRIMITIVES[name]
        return "void" if ret and s == "undefined" else s
    if name == "optional":
        return f"{ts_type(t['optional_type'])} | undefined"
    if name == "variant":
        return " | ".join(ts_type(v) for v in t["variant_types"])
    if name == "array":
        inner = ts_type(t["element_type"])
        return f"({inner})[]" if " " in inner else f"{inner}[]"
    if name == "map":
        return f"Record<{ts_type(t['key_type'])}, {ts_type(t['value_type'])}>"
    if name == "promise":
        return f"Promise<{ts_type(t['promise_type'])}>"
    if name == "generator":
        g = t["generator_type"]
        return f"Generator<{ts_type(g['yield_type'])}, {ts_type(g['return_type'])}, {ts_type(g['next_type'])}>"
    if name == "closure":
        c = t["closure_type"]
        args = ", ".join(f"arg{i}: {ts_type(a)}" for i, a in enumerate(c["argument_types"]))
        return f"({args}) => {ts_type(c['return_type'], ret=True)}"
    return name  # クラス・インターフェイス・ジェネリクス等はそのまま


def fmt_args(fn: dict) -> str:
    out = []
    for a in fn.get("arguments", []):
        t = a["type"]
        if t["name"] == "optional":
            out.append(f"{a['name']}?: {ts_type(t['optional_type'])}")
        else:
            out.append(f"{a['name']}: {ts_type(t)}")
    return ", ".join(out)


def fmt_throws(t: dict) -> list[str]:
    errs = [e["name"] for e in t.get("error_types", [])]
    return [f"    /** @throws {', '.join(errs)} */"] if errs else []


def render_property(p: dict, indent: str = "    ") -> str:
    ro = "readonly " if p.get("is_read_only") else ""
    st = "static " if p.get("is_static") else ""
    t = p["type"]
    if t["name"] == "optional":
        return f"{indent}{st}{ro}{p['name']}?: {ts_type(t['optional_type'])};"
    return f"{indent}{st}{ro}{p['name']}: {ts_type(t)};"


def render_class(c: dict, keyword: str = "class") -> list[str]:
    ext = f" extends {', '.join(b['name'] for b in c['base_types'])}" if c.get("base_types") else ""
    lines = [f"export {keyword} {c['name']}{ext} {{"]
    for k in c.get("constants", []):
        lines.append(f"    static readonly {k['name']} = {json.dumps(k['value'])};")
    for p in c.get("properties", []):
        lines += fmt_throws(p["type"])
        lines.append(render_property(p))
    for f in c.get("functions", []):
        if f.get("is_constructor"):
            lines.append(f"    constructor({fmt_args(f)});")
            continue
        lines += fmt_throws(f["return_type"])
        st = "static " if f.get("is_static") else ""
        lines.append(f"    {st}{f['name']}({fmt_args(f)}): {ts_type(f['return_type'], ret=True)};")
    lines.append("}")
    return lines


def render_module(d: dict, version: str) -> str:
    L = [f"// {d['name']} {version}"]
    for dep in d.get("peer_dependencies", []):
        if not dep.get("types_only"):
            alias = re.sub(r"[^a-z]", "", dep["name"].split("/")[-1])
            L.append(f'import * as minecraft{alias} from "{dep["name"]}";')
    for e in d.get("enums", []):
        L.append(f"export enum {e['name']} {{")
        L += [f"    {k['name']} = {json.dumps(k['value'])}," for k in e["constants"]]
        L.append("}")
    for a in d.get("type_aliases", []):
        if a.get("raw_script_text"):
            L.append(a["raw_script_text"])
        elif a.get("alias_type") == "type_map":
            L.append(f"export type {a['name']} = {{")
            L += [f"    '{m['name']}': {m['value']};" for m in a["mappings"]]
            L.append("};")
    for c in d.get("classes", []):
        L += render_class(c)
    for i in d.get("interfaces", []):
        L += render_class(i, "interface")
    for e in d.get("errors", []):
        L += render_class({**e, "base_types": [{"name": "Error"}]})
    for f in d.get("functions", []):
        L.append(f"export function {f['name']}({fmt_args(f)}): {ts_type(f['return_type'], ret=True)};")
    for k in d.get("constants", []):
        L.append(f"export const {k['name']} = {json.dumps(k.get('value'))};")
    for o in d.get("objects", []):
        L.append(f"export const {o['name']}: {ts_type(o['type'])};")
    return "\n".join(L) + "\n"


def _vkey(path: Path) -> tuple:
    ver = path.stem.split("_", 1)[1]
    nums = tuple(int(x) for x in re.findall(r"\d+", ver.split("-")[0]))
    return ("-beta" not in ver, nums)  # 安定版を優先し、その中で最新


# 学習に使うモジュール: 実体(bindings)と、利用者がimportする公開名の対応
MODULES = {
    "server-bindings": "@minecraft/server",
    "server-ui-bindings": "@minecraft/server-ui",
    "server-admin": "@minecraft/server-admin",
    "server-net": "@minecraft/server-net",
    "common": "@minecraft/common",
}


def latest_module_files(script_dir: Path) -> dict[str, Path]:
    out = {}
    for base, public in MODULES.items():
        files = [p for p in script_dir.glob(f"{base}_*.json") if p.stem.split("_", 1)[0] == base]
        if files:
            out[public] = max(files, key=_vkey)
    return out


def render_all(script_dir: Path) -> dict[str, str]:
    """{公開モジュール名: d.ts文字列}"""
    result = {}
    for public, path in latest_module_files(script_dir).items():
        d = json.loads(path.read_text(encoding="utf-8"))
        d["name"] = public
        result[public] = render_module(d, path.stem.split("_", 1)[1])
    return result


if __name__ == "__main__":
    import sys

    root = Path(sys.argv[1]) / "metadata" / "script_modules" / "@minecraft"
    for name, text in render_all(root).items():
        print(f"{name}: {len(text):,} chars")
    print(render_all(root)["@minecraft/server"][:3000])
