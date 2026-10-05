"""Script API（JS/TS）を公式型定義（@minecraft/server 等）で型チェックする。

    from bedrock_tools.script_check import check_scripts
    issues = check_scripts({"scripts/main.js": "import { world } from '@minecraft/server'; ..."})

script_check/ に typescript と型定義が npm install 済みであること（README参照）。
strict ではなく、存在しないAPI・誤った引数型・存在しないモジュールの検出を主目的にしている。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from bedrock_tools.schema_validator import PROJECT_ROOT

CHECK_DIR = PROJECT_ROOT / "script_check"
TSC = CHECK_DIR / "node_modules" / "typescript" / "bin" / "tsc"
ERROR_RE = re.compile(r"^(?P<file>.+?)\((?P<line>\d+),(?P<col>\d+)\): error (?P<code>TS\d+): (?P<msg>.*)$")


@dataclass
class ScriptIssue:
    file: str
    line: int
    col: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}:{self.col} {self.code} {self.message}"


def available() -> bool:
    return TSC.exists() and shutil.which("node") is not None


def _js_false_positive(issue: ScriptIssue) -> bool:
    """JS では問題なく動くのに型チェックだけが落ちるもの。
    getState / withState の引数はバニラのステート名（keyof BlockStateSuperset）に型付けされているため、
    カスタムブロックのステート（"my:lit" など）がエラーになる。TS では `as keyof BlockStateSuperset` 等が必要なので残す。"""
    return issue.file.endswith(".js") and "BlockStateSuperset" in issue.message


def check_scripts(files: dict[str, str], timeout: int = 120) -> list[ScriptIssue]:
    """files: {相対パス: 内容}（.js / .ts）。型チェックのエラー一覧を返す（空なら合格）。"""
    if not available():
        raise RuntimeError("script_check の準備ができていません: cd script_check && npm install")
    work = CHECK_DIR / "work" / uuid.uuid4().hex[:12]
    try:
        for rel, text in files.items():
            dest = work / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8", newline="\n")
        (work / "tsconfig.json").write_text(json.dumps({
            "extends": "../../tsconfig.base.json",
            "include": list(files),
            "files": ["../../globals.d.ts"],  # console 等、Minecraftの実行環境にあるグローバル
        }), encoding="utf-8")
        res = subprocess.run(["node", str(TSC), "-p", str(work / "tsconfig.json"), "--pretty", "false"],
                             capture_output=True, text=True, encoding="utf-8", timeout=timeout)
        issues = []
        for line in res.stdout.splitlines():
            m = ERROR_RE.match(line.strip())
            if m:
                f = Path(m["file"])
                try:
                    f = f.resolve().relative_to(work.resolve())
                except ValueError:
                    pass
                issues.append(ScriptIssue(f.as_posix(), int(m["line"]), int(m["col"]), m["code"], m["msg"]))
            elif line.strip() and issues:  # 複数行メッセージの続き
                issues[-1].message += " " + line.strip()
        return [i for i in issues if not _js_false_positive(i)]
    finally:
        shutil.rmtree(work, ignore_errors=True)
