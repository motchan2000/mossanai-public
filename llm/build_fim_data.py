"""FIM（穴埋め補完）学習データの作成。VSCodeでのインライン補完用。

trainに分けたvanillaファイル（元の書式のまま）から、カーソル位置を模した穴を作る:
  line_rest … 行の途中から行末まで（タイピング中の補完）
  line      … 1行まるごと
  block     … コンポーネント1つ分（複数行）
  span      … ランダムな範囲（汎化用）
各サンプルは middle の後に <|endoftext|> を付けて「書き終えたら止まる」ことも学習させる。

出力: data/sft/fim_train.jsonl  {"prompt": ..., "completion": ...}
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from llm.fim import PROJECT_ROOT, fim_prompt
from llm.fim_tasks import BLOCK_START_RE, LINE_RE, _block_end, is_val

DIRS = {  # 相対パス接頭辞: 1ファイルあたりのサンプル数
    "behavior_pack/entities/": 12, "behavior_pack/items/": 4, "behavior_pack/blocks/": 4,
    "behavior_pack/spawn_rules/": 4, "behavior_pack/loot_tables/": 3, "behavior_pack/recipes/": 1,
    "behavior_pack/trading/": 4, "behavior_pack/biomes/": 2, "behavior_pack/": 1,
    "resource_pack/entity/": 6, "resource_pack/animation_controllers/": 6, "resource_pack/animations/": 4,
    "resource_pack/render_controllers/": 3, "resource_pack/attachables/": 3, "resource_pack/particles/": 3,
    "resource_pack/models/": 2, "resource_pack/biomes/": 1, "resource_pack/textures/": 0.1,
    "resource_pack/sounds": 4,
}
SKIP = ("resource_pack/ui/", "resource_pack/texts/")
EOT = "<|endoftext|>"


def per_file(rel: str) -> float:
    for d, n in DIRS.items():
        if rel.startswith(d):
            return n
    return 1


def make_samples(text: str, rel: str, rng: random.Random, n: int,
                 max_prefix: int = 4000, max_suffix: int = 1500) -> list[dict]:
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for l in lines:
        offsets.append(offsets[-1] + len(l))
    line_idx = [i for i, l in enumerate(lines) if LINE_RE.match(l.rstrip("\n"))]
    blocks = []
    for i, l in enumerate(lines):
        if BLOCK_START_RE.match(l.rstrip("\n")):
            j = _block_end(lines, i)
            if j is not None and 1 <= j - i <= 40:
                blocks.append((i, j))

    out = []
    for _ in range(n):
        kind = rng.choices(["line_rest", "line", "block", "span"], weights=[4, 2, 3, 1])[0]
        if kind in ("line_rest", "line") and line_idx:
            i = rng.choice(line_idx)
            body = lines[i].rstrip("\n")
            ind = len(body) - len(body.lstrip())
            start = offsets[i] + (rng.randint(ind, max(ind, len(body) - 1)) if kind == "line_rest" else ind)
            end = offsets[i] + len(body)
        elif kind == "block" and blocks:
            i, j = rng.choice(blocks)
            ind = len(lines[i]) - len(lines[i].lstrip())
            start = offsets[i] + ind
            end = offsets[j] + len(lines[j].rstrip("\n").rstrip().rstrip(","))
        else:
            kind = "span"
            start = rng.randint(0, max(0, len(text) - 2))
            end = min(len(text), start + rng.randint(5, 400))
        if end <= start:
            continue
        prefix, middle, suffix = text[:start][-max_prefix:], text[start:end], text[end:][:max_suffix]
        # ファイルパス付き（自作CLI用）と無し（Continue等の既定テンプレート互換）を混ぜる
        path = rel if rng.random() < 0.3 else None
        out.append({"prompt": fim_prompt(prefix, suffix, path), "completion": middle + EOT, "kind": kind,
                    "path": rel})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "sft" / "fim_train.jsonl")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0, help="サンプル数の倍率")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    a = ap.parse_args()
    if a.split == "val" and a.out.name == "fim_train.jsonl":
        a.out = a.out.with_name("fim_val.jsonl")

    src = sorted((PROJECT_ROOT / "data" / "bedrock" / "raw").glob("bedrock-samples-*"))[-1]
    rng = random.Random(a.seed)
    samples = []
    for p in sorted(src.rglob("*.json")):
        rel = p.relative_to(src).as_posix()
        if rel.startswith(SKIP) or rel.startswith("metadata/") or is_val(rel) != (a.split == "val"):
            continue
        n = per_file(rel) * a.scale
        n = int(n) + (rng.random() < n - int(n))
        if n:
            samples += make_samples(p.read_text(encoding="utf-8", errors="replace"), rel, rng, n)
    rng.shuffle(samples)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(len(samples), Counter(s["kind"] for s in samples))
    print(Counter("/".join(s["path"].split("/")[:2]) for s in samples).most_common(12))


if __name__ == "__main__":
    main()
