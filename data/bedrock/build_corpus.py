"""STEP9: bedrock-samples から構造タグ付き学習コーパスを作る。

出力 (data/bedrock/corpus/):
    train.txt / val.txt  … 1ドキュメント = "<TAG>本文</TAG>\n"
    stats.json           … タグ別のドキュメント数・文字数
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

here = Path(__file__).parent
sys.path.insert(0, str(here.parent.parent))  # プロジェクトルート（bedrock_tools）

from bedrock_tools.jsonc import dump as dump_json, parse_lenient, strip_json_comments  # noqa: E402,F401
from script_api import render_all  # noqa: E402

DEFAULT_SRC = here / "raw" / "bedrock-samples-v1.26.50.4"

# (パス接頭辞, タグ)。上から順に最初にマッチしたものを使う。
PATH_TAGS = [
    ("behavior_pack/manifest.json", "MANIFEST"),
    ("resource_pack/manifest.json", "MANIFEST"),
    ("behavior_pack/entities/", "BP_ENTITY"),
    ("behavior_pack/items/", "BP_ITEM"),
    ("behavior_pack/blocks/", "BP_BLOCK"),
    ("behavior_pack/recipes/", "BP_RECIPE"),
    ("behavior_pack/loot_tables/", "BP_LOOT"),
    ("behavior_pack/spawn_rules/", "BP_SPAWN"),
    ("behavior_pack/trading/", "BP_TRADING"),
    ("behavior_pack/biomes/", "BP_BIOME"),
    ("behavior_pack/features/", "BP_FEATURE"),
    ("behavior_pack/feature_rules/", "BP_FEATURE"),
    ("behavior_pack/", "BP_OTHER"),
    ("resource_pack/entity/", "RP_ENTITY"),
    ("resource_pack/models/", "RP_MODEL"),
    ("resource_pack/animations/", "RP_ANIM"),
    ("resource_pack/animation_controllers/", "RP_ANIM_CTRL"),
    ("resource_pack/render_controllers/", "RP_RENDER_CTRL"),
    ("resource_pack/particles/", "RP_PARTICLE"),
    ("resource_pack/attachables/", "RP_ATTACHABLE"),
    ("resource_pack/sounds", "RP_SOUND"),
    ("resource_pack/textures/", "RP_TEXTURE"),
    ("resource_pack/ui/", None),  # スコープ外
    ("resource_pack/texts/", None),  # 翻訳ファイル（スコープ外）
    ("resource_pack/", "RP_OTHER"),
]

ALL_TAGS = sorted({t for _, t in PATH_TAGS if t} | {"SCRIPT_TS", "SCRIPT_DTS", "DOC", "MOLANG_DOC"})


def tag_for(rel: str) -> str | None:
    for prefix, tag in PATH_TAGS:
        if rel.startswith(prefix):
            return tag
    return None


def parse_json(text: str):
    """コメント・末尾カンマを許容してパース。失敗時はNone。"""
    return parse_lenient(text)[0]


def normalize_json(text: str) -> tuple[str, bool]:
    """2スペースインデントに正規化。パースできなければ元テキストを返す。"""
    obj = parse_json(text)
    if obj is None:
        return text.lstrip("﻿").strip(), False
    return dump_json(obj), True


def split_large_json(obj, max_chars: int) -> list:
    """巨大なJSON（sound_definitions.json等）を、最大の辞書フィールドのエントリ単位で分割する。

    各断片は「その他のフィールド + 分割したエントリの一部」を持つ有効なJSONになる。
    1ファイルがval全体を占めてしまう偏りや、context長を大きく超える問題を防ぐ。
    """
    total = len(dump_json(obj))
    if total <= max_chars or not isinstance(obj, dict):
        return [obj]

    # 分割対象の辞書を決める: 子がサイズの過半を占める間だけ降りる（最大2段）。
    # 例: sound_definitions.json → obj["sound_definitions"]、blocks.json → obj自身
    path: tuple[str, ...] = ()
    target = obj
    while len(path) < 2:
        sizes = {k: len(dump_json(v)) for k, v in target.items() if isinstance(v, dict)}
        if not sizes:
            break
        k = max(sizes, key=sizes.get)
        if sizes[k] < 0.5 * len(dump_json(target)):
            break
        path, target = path + (k,), target[k]
    if len(target) < 2:
        return [obj]

    def small(v) -> bool:
        return len(json.dumps(v)) < 200

    def with_entries(entries: dict):
        # パス上の小さいフィールド(format_version, description等)だけを残す（大きい兄弟は複製しない）
        if not path:
            return entries
        out = {k: v for k, v in obj.items() if small(v)}
        src, dst = obj, out
        for k in path[:-1]:
            src = src[k]
            dst[k] = {kk: vv for kk, vv in src.items() if small(vv)}
            dst = dst[k]
        dst[path[-1]] = entries
        return out

    # 分割対象を除いた残り（大きい兄弟フィールド）は別ドキュメントとして再帰的に分割。
    # 対象は過半を占めているので、残りは必ず元の半分未満 → 再帰は必ず縮小して終わる。
    chunks = []
    if path:
        rest = json.loads(json.dumps(obj))
        node = rest
        for k in path[:-1]:
            node = node[k]
        del node[path[-1]]
        if len(dump_json(rest)) - len(dump_json(with_entries({}))) > 500:
            chunks = split_large_json(rest, max_chars)

    cur, cur_len = {}, 0
    for k, v in target.items():
        size = len(dump_json({k: v}))
        if cur and cur_len + size > max_chars:
            chunks.append(with_entries(cur))
            cur, cur_len = {}, 0
        cur[k] = v
        cur_len += size
    if cur:
        chunks.append(with_entries(cur))
    return chunks


# ---------------------------------------------------------------------------
# 公式ドキュメント(doc_modules) → Markdown
# ---------------------------------------------------------------------------
def render_doc_node(node: dict, level: int) -> list[str]:
    L = []
    name = node.get("name", "")
    if node.get("type") is not None or node.get("default") is not None:
        # テーブル行として扱われる「プロパティ定義」
        meta = ", ".join(x for x in [node.get("type") and str(node["type"]),
                                     node.get("default") not in (None, "") and f"default: {node['default']}"] if x)
        desc = " ".join(s.strip() for s in node.get("description", []) if s.strip())
        L.append(f"- `{name}`" + (f" ({meta})" if meta else "") + (f": {desc}" if desc else ""))
    else:
        if name:
            L.append(f"{'#' * min(level, 6)} {name}")
        L += [s.rstrip() for s in node.get("description", [])]
    for ex in node.get("examples", []):
        if ex.get("name"):
            L.append(f"Example: {ex['name']}")
        L += ["```", *[t.replace("\t", "  ") for t in ex.get("text", [])], "```"]
    for child in node.get("nodes", []):
        L += render_doc_node(child, level + 1)
    return L


def chunk_markdown(lines: list[str], max_chars: int) -> list[list[str]]:
    """見出し行の位置で区切り、max_chars以下の塊にまとめる。各塊の先頭に直近のH1/H2見出しを付ける。"""
    chunks, cur, cur_len, heads = [], [], 0, {}
    for line in lines:
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            if cur and cur_len + len(line) > max_chars:
                chunks.append(cur)
                cur = [heads[l] for l in sorted(heads) if l < level and l <= 2]
                cur_len = sum(map(len, cur))
            heads = {l: h for l, h in heads.items() if l < level}
            heads[level] = line
        elif cur and cur_len + len(line) > max_chars * 1.5:
            # 見出しの無い長大なリスト（AuxValue表など）は行単位で強制分割
            chunks.append(cur)
            cur = [heads[l] for l in sorted(heads) if l <= 2]
            cur_len = sum(map(len, cur))
        cur.append(line)
        cur_len += len(line) + 1
    if cur:
        chunks.append(cur)
    return chunks


def doc_documents(meta_dir: Path, max_chars: int) -> list[tuple[str, str, str]]:
    docs = []
    for f in sorted((meta_dir / "doc_modules").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for i, node in enumerate(d["nodes"]):
            for j, chunk in enumerate(chunk_markdown(render_doc_node(node, 1), max_chars)):
                body = "\n".join(chunk).strip()
                if body:
                    docs.append(("DOC", f"doc:{f.stem}:{i}:{j}", f"[{d['name']}]\n{body}"))
    return docs


def molang_documents(meta_dir: Path) -> list[tuple[str, str, str]]:
    q = json.loads((meta_dir / "molang_modules" / "mojang-molang-queries.json").read_text(encoding="utf-8"))
    docs = []
    for kind in ("queries", "math_functions"):
        for item in q[kind]:
            args = f"args {item.get('min_args', 0)}-{item.get('max_args', item.get('min_args', 0))}"
            ret = f" -> {item['return_type']}" if item.get("return_type") else ""
            line = f"{item['name']} ({args}){ret}: {item.get('description', '').strip()}"
            docs.append(("MOLANG_DOC", f"molang:{item['name']}", line))
    return docs


def script_api_documents(meta_dir: Path) -> list[tuple[str, str, str]]:
    """d.tsをトップレベル宣言ごとに分割（1モジュール丸ごとだとcontext長を大きく超えるため）。"""
    docs = []
    for module, text in render_all(meta_dir / "script_modules" / "@minecraft").items():
        lines = text.splitlines()
        header = [l for l in lines[:5] if l.startswith(("//", "import"))]
        chunk: list[str] = []
        for line in lines[len(header):]:
            if line.startswith("export ") and chunk:
                docs.append(("SCRIPT_DTS", f"dts:{module}:{len(docs)}", "\n".join(header + chunk)))
                chunk = []
            chunk.append(line)
        if chunk:
            docs.append(("SCRIPT_DTS", f"dts:{module}:{len(docs)}", "\n".join(header + chunk)))
    return docs


def pack_documents(src: Path, max_chars: int) -> tuple[list[tuple[str, str, str]], int]:
    docs, broken = [], 0
    for p in sorted(src.rglob("*")):
        if not p.is_file() or ".git" in p.parts:
            continue
        rel = p.relative_to(src).as_posix()
        tag = tag_for(rel)
        if tag is None:
            continue
        if p.suffix == ".json":
            text = p.read_text(encoding="utf-8", errors="replace")
            obj = parse_json(text)
            if obj is None:
                broken += 1
                docs.append((tag, rel, text.lstrip("﻿").strip()))
                continue
            chunks = split_large_json(obj, max_chars)
            for i, chunk in enumerate(chunks):
                docs.append((tag, rel if len(chunks) == 1 else f"{rel}#{i}", dump_json(chunk)))
        elif p.suffix in (".js", ".ts"):
            docs.append(("SCRIPT_TS", rel, p.read_text(encoding="utf-8").strip()))
    return docs, broken


def is_val(key: str, ratio: float) -> bool:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < ratio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=here / "corpus")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--max_doc_chars", type=int, default=20000, help="これを超えるJSONは分割する")
    a = ap.parse_args()

    pack, broken = pack_documents(a.src, a.max_doc_chars)
    meta = a.src / "metadata"
    docs = pack + script_api_documents(meta) + doc_documents(meta, a.max_doc_chars) + molang_documents(meta)

    seen, uniq = set(), []
    for tag, key, body in docs:
        h = hashlib.md5(body.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq.append((tag, key, body))

    a.out.mkdir(parents=True, exist_ok=True)
    count, chars = Counter(), Counter()
    with open(a.out / "train.txt", "w", encoding="utf-8", newline="\n") as ft, \
         open(a.out / "val.txt", "w", encoding="utf-8", newline="\n") as fv:
        for tag, key, body in uniq:
            split = "val" if is_val(key, a.val_ratio) else "train"
            (fv if split == "val" else ft).write(f"<{tag}>{body}</{tag}>\n")
            count[(tag, split)] += 1
            chars[tag] += len(body)

    stats = {
        "source": a.src.name, "documents": len(uniq), "duplicates_removed": len(docs) - len(uniq),
        "json_parse_failed": broken,
        "by_tag": {t: {"train": count[(t, "train")], "val": count[(t, "val")], "chars": chars[t]}
                   for t in sorted(chars, key=lambda t: -chars[t])},
    }
    (a.out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
