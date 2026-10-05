"""コマンドライン:

    uv run python -m bedrock_tools check <パックのフォルダ or ファイル> [-v]
    uv run python -m bedrock_tools new "<アドオン名>" [--description ...] [--script] [--out <出力先>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="bedrock_tools")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="アドオンのJSONを検証する")
    c.add_argument("path", type=Path)
    c.add_argument("-v", "--verbose", action="store_true", help="警告も表示")

    n = sub.add_parser("new", help="BP/RPの雛形を作る")
    n.add_argument("name")
    n.add_argument("--description", default="")
    n.add_argument("--script", action="store_true", help="Script API (scripts/main.js) を有効化")
    n.add_argument("--out", type=Path, default=Path("."))

    a = ap.parse_args(argv)
    if a.cmd == "check":
        from bedrock_tools.check import check_path, format_report

        report = check_path(a.path)
        print(format_report(report, a.path, a.verbose))
        return 1 if report.error_count else 0
    if a.cmd == "new":
        from bedrock_tools.manifest import scaffold

        bp, rp = scaffold(a.out, a.name, a.description, a.script)
        print(f"作成しました:\n  {bp}\n  {rp}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
