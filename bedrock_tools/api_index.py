"""公式型定義（npm の @minecraft/server 等の index.d.ts）から、プロンプトに入れるためのAPI索引を作る。

- 宣言（class / interface / enum / type）単位に分割
- JSDoc は1行の要約に圧縮し、メンバーの横にコメントとして残す（モデルが用途を判断できるように）
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from bedrock_tools.script_check import CHECK_DIR

MODULES = {"@minecraft/server": "server", "@minecraft/server-ui": "server-ui"}
DECL_RE = re.compile(r"^export (?:declare )?(?:abstract )?(class|interface|enum|type) (\w+)", re.M)


@dataclass
class Decl:
    module: str
    kind: str
    name: str
    text: str  # 圧縮した宣言

    @property
    def size(self) -> int:
        return len(self.text)


def _summary(doc: str) -> str:
    """JSDoc から @remarks の最初の文だけ取り出す。"""
    body = re.sub(r"^\s*/\*\*|\*/\s*$", "", doc)
    lines = [re.sub(r"^\s*\*\s?", "", l) for l in body.splitlines()]
    text, in_remarks = [], False
    for l in lines:
        if l.startswith("@remarks"):
            in_remarks = True
            l = l[len("@remarks"):].strip()
        elif l.startswith("@"):
            in_remarks = False
            continue
        if in_remarks and l.strip():
            text.append(l.strip())
        elif in_remarks and text:
            break
    s = " ".join(text)
    s = s.split(". ")[0].rstrip(".")
    return s[:120]


def _compact(block: str) -> str:
    """宣言ブロックを、JSDoc要約付きのメンバー一覧に圧縮する。"""
    out, pending = [], ""
    # 先頭の空白も含めて JSDoc を先に試す（でないとインデントの行が通常行として先に一致してしまう）
    for m in re.finditer(r"([ \t]*/\*\*.*?\*/)|([^\n]+)", block, re.S):
        if m.group(1):
            pending = _summary(m.group(1))
            continue
        line = m.group(2).rstrip()
        if not line.strip() or line.strip().startswith("//") or "private constructor" in line:
            continue
        if line.strip().startswith(("*", "/*")):
            continue
        if pending and line.strip() not in ("{", "}") and not line.startswith("export"):
            line = f"{line}  // {pending}"
        pending = ""
        out.append(line)
    return "\n".join(out)


@lru_cache(maxsize=1)
def load_index() -> dict[str, Decl]:
    index: dict[str, Decl] = {}
    for module, folder in MODULES.items():
        text = (CHECK_DIR / "node_modules" / "@minecraft" / folder / "index.d.ts").read_text(encoding="utf-8")
        starts = [m for m in DECL_RE.finditer(text)]
        for i, m in enumerate(starts):
            # 宣言の直前の JSDoc は捨て、次の宣言の手前（その JSDoc の開始）までを本体とする
            end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
            chunk = text[m.start():end]
            doc_start = chunk.rfind("/**")
            if doc_start > 0 and chunk[:doc_start].rstrip().endswith(("}", ";")):
                chunk = chunk[:doc_start]
            name = m.group(2)
            if name not in index or module == "@minecraft/server":
                index[name] = Decl(module, m.group(1), name, _compact(chunk.strip()))
    return index


def signature(name: str, max_chars: int = 6000) -> str | None:
    d = load_index().get(name)
    if not d:
        return None
    text = d.text if d.size <= max_chars else d.text[:max_chars] + "\n    // …（省略）\n}"
    return f"// from \"{d.module}\"\n{text}"


def event_table() -> str:
    """world / system の afterEvents・beforeEvents の一覧（名前と1行説明）。"""
    lines = []
    for cls, path in (("WorldAfterEvents", "world.afterEvents"), ("WorldBeforeEvents", "world.beforeEvents"),
                      ("SystemAfterEvents", "system.afterEvents"), ("SystemBeforeEvents", "system.beforeEvents")):
        d = load_index().get(cls)
        if not d:
            continue
        for m in re.finditer(r"readonly (\w+): (\w+);(?:\s*// (.*))?", d.text):
            lines.append(f"{path}.{m.group(1)} → {m.group(2)}" + (f"  // {m.group(3)}" if m.group(3) else ""))
    return "\n".join(lines)


def event_payload(signal: str) -> str | None:
    """XxxAfterEventSignal → XxxAfterEvent のように、イベント引数の型名を返す。"""
    base = signal.removesuffix("Signal")
    return base if base in load_index() else None


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    idx = load_index()
    print(len(idx), "decls;", sum(d.size for d in idx.values()) // 1000, "k chars")
    print(signature("EntityDieAfterEvent"))
    print(signature("ActionFormData")[:1500])
    print(event_table()[:1500])
