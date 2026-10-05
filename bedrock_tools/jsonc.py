"""Bedrockのゆるい JSON（コメント・末尾カンマ・BOM）を扱うユーティリティ。"""
from __future__ import annotations

import json
import re


def strip_json_comments(text: str) -> str:
    """文字列リテラルの外にある // と /* */ を除去（Bedrockのjsonはコメントを許容するため）。"""
    out, i, n, in_str = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            out.append(c)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        else:
            out.append(c)
        i += 1
    return "".join(out)


def parse_lenient(text: str):
    """コメント・末尾カンマを許容してパース。(obj, None) または (None, エラーメッセージ)。"""
    text = strip_json_comments(text.lstrip("﻿"))
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        first = e
    try:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", text)), None
    except json.JSONDecodeError:
        return None, f"JSON構文エラー: {first.msg} (行 {first.lineno}, 列 {first.colno})"


def dump(obj, indent: int = 2, inline_max: int = 80) -> str:
    """プロジェクト標準の整形: 2スペースインデント。
    スカラーだけの短い配列（"version": [1, 0, 0] や座標など）は1行にまとめる（人が書くBedrock JSONの慣習）。"""

    def fmt(v, level: int) -> str:
        pad, pad_in = " " * (indent * level), " " * (indent * (level + 1))
        if isinstance(v, dict):
            if not v:
                return "{}"
            items = [f"{pad_in}{json.dumps(k, ensure_ascii=False)}: {fmt(x, level + 1)}" for k, x in v.items()]
            return "{\n" + ",\n".join(items) + f"\n{pad}}}"
        if isinstance(v, list):
            if not v:
                return "[]"
            if all(not isinstance(x, (dict, list)) for x in v):
                one_line = "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in v) + "]"
                if len(one_line) <= inline_max:
                    return one_line
            return "[\n" + ",\n".join(f"{pad_in}{fmt(x, level + 1)}" for x in v) + f"\n{pad}]"
        return json.dumps(v, ensure_ascii=False)

    return fmt(obj, 0)
