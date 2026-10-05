"""FIM評価タスクの作成（VSCode補完を想定）。

- line : "key": value の行を、インデントの直後から行末まで隠す
- block: "minecraft:xxx": { ... } のコンポーネント1つ分を隠す
決定的（seed固定）に作るので、ベースモデルと学習後モデルを同じ問題で比較できる。
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from llm.fim import PROJECT_ROOT

LINE_RE = re.compile(r'^(\s*)("[^"]+"\s*:\s*[^\s{\[].*?)\s*$')
BLOCK_START_RE = re.compile(r'^(\s*)("minecraft:[^"]+"\s*:\s*\{)\s*$')

# 評価に使うファイル（相対パスの接頭辞）
EVAL_DIRS = [
    "behavior_pack/entities/", "behavior_pack/items/", "behavior_pack/blocks/",
    "behavior_pack/spawn_rules/", "behavior_pack/loot_tables/", "behavior_pack/recipes/",
    "resource_pack/entity/", "resource_pack/animation_controllers/", "resource_pack/render_controllers/",
    "resource_pack/attachables/",
]


@dataclass
class FimTask:
    id: str
    kind: str  # line | block
    path: str
    prefix: str
    suffix: str
    expected: str

    def full(self, middle: str) -> str:
        return self.prefix + middle + self.suffix


def is_val(rel: str, ratio: float = 0.05) -> bool:
    """build_corpus.py と同じ train/val 分割（学習時にvalのファイルを使わないため）。"""
    h = int(hashlib.md5(rel.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < ratio


def _block_end(lines: list[str], start: int) -> int | None:
    depth = 0
    for i in range(start, len(lines)):
        # 文字列内の括弧は稀なので簡易カウント（評価用の問題作成にのみ使用）
        depth += lines[i].count("{") - lines[i].count("}")
        if depth == 0:
            return i
    return None


def make_tasks(text: str, rel: str, rng: random.Random, n_line: int, n_block: int,
               max_prefix_chars: int = 6000, max_suffix_chars: int = 2000) -> list[FimTask]:
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for l in lines:
        offsets.append(offsets[-1] + len(l))
    tasks = []

    def clip(prefix: str, suffix: str) -> tuple[str, str]:
        return prefix[-max_prefix_chars:], suffix[:max_suffix_chars]

    line_cands = [i for i, l in enumerate(lines) if LINE_RE.match(l.rstrip("\n"))]
    for i in rng.sample(line_cands, min(n_line, len(line_cands))):
        m = LINE_RE.match(lines[i].rstrip("\n"))
        start = offsets[i] + len(m.group(1))
        end = start + len(m.group(2))
        p, s = clip(text[:start], text[end:])
        tasks.append(FimTask(f"{rel}#L{i}", "line", rel, p, s, m.group(2)))

    block_cands = []
    for i, l in enumerate(lines):
        if BLOCK_START_RE.match(l.rstrip("\n")):
            end = _block_end(lines, i)
            if end is not None and 1 <= end - i <= 25:
                block_cands.append((i, end))
    for i, j in rng.sample(block_cands, min(n_block, len(block_cands))):
        m = BLOCK_START_RE.match(lines[i].rstrip("\n"))
        start = offsets[i] + len(m.group(1))
        end_line = lines[j].rstrip("\n").rstrip()
        end = offsets[j] + len(end_line.rstrip(","))  # 閉じ括弧まで（後続のカンマは残す）
        p, s = clip(text[:start], text[end:])
        tasks.append(FimTask(f"{rel}#B{i}", "block", rel, p, s, text[start:end]))
    return tasks


def build_eval_set(src: Path, seed: int = 0, n_line: int = 3, n_block: int = 2,
                   max_per_dir: int = 60) -> list[FimTask]:
    rng = random.Random(seed)
    tasks = []
    for d in EVAL_DIRS:
        per_dir = []
        for p in sorted(src.glob(d + "**/*.json")):
            rel = p.relative_to(src).as_posix()
            if is_val(rel):
                per_dir += make_tasks(p.read_text(encoding="utf-8"), rel, rng, n_line, n_block)
        if len(per_dir) > max_per_dir:  # recipes等の件数が多い種類に偏らないように
            per_dir = rng.sample(per_dir, max_per_dir)
        tasks += per_dir
    return tasks


def default_eval_path() -> Path:
    return PROJECT_ROOT / "data" / "eval" / "fim_val.jsonl"


def save(tasks: list[FimTask], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def load_tasks(path: Path) -> list[FimTask]:
    return [FimTask(**json.loads(l)) for l in path.read_text(encoding="utf-8").splitlines() if l]


if __name__ == "__main__":
    src = sorted((PROJECT_ROOT / "data" / "bedrock" / "raw").glob("bedrock-samples-*"))[-1]
    tasks = build_eval_set(src)
    save(tasks, default_eval_path())
    from collections import Counter

    print(len(tasks), Counter(t.kind for t in tasks), Counter(t.path.split("/")[1] for t in tasks))
