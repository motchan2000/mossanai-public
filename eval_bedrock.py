"""Bedrockモデルの評価（設計図「2. 評価指標」の一部）。

1. タグ別 val loss（どの種類のファイルが苦手かを把握する）
2. JSON構文妥当性率: 各タグの開始トークンから生成し、json.loadsで通る割合
   （--json_guard 有り/無しを比較）
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from sampling import BracketGuard, generate
from tokenizer_utils import load_tokenizer

JSON_TAGS = ["BP_ENTITY", "BP_ITEM", "BP_BLOCK", "BP_RECIPE", "BP_LOOT", "BP_SPAWN",
             "RP_ENTITY", "RP_MODEL", "RP_ANIM", "RP_ANIM_CTRL", "RP_RENDER_CTRL", "RP_PARTICLE"]


@torch.no_grad()
def per_tag_loss(model: GPT, tok, val_text: str, device: str) -> dict[str, dict]:
    T = model.cfg.block_size
    stats = defaultdict(lambda: [0.0, 0])
    for m in re.finditer(r"<([A-Z_]+)>.*?</\1>\n", val_text, re.S):
        ids = torch.tensor(tok.encode(m.group(0)), device=device)
        # 長いドキュメントは block_size ごとに区切って評価
        for s in range(0, len(ids) - 1, T):
            x = ids[s : s + T][None]
            y = ids[s + 1 : s + 1 + T][None]
            x = x[:, : y.size(1)]
            with torch.autocast(device, dtype=torch.bfloat16):
                logits, _ = model(x, targets=y)
            stats[m.group(1)][0] += F.cross_entropy(logits[0].float(), y[0], reduction="sum").item()
            stats[m.group(1)][1] += y.size(1)
    return {t: {"loss": round(l / n, 3), "tokens": n} for t, (l, n) in sorted(stats.items(), key=lambda x: -x[1][1])}


def json_validity(model: GPT, tok, device: str, n: int, guard: bool, max_new: int) -> dict:
    res = {}
    for tag in JSON_TAGS:
        ok = 0
        for i in range(n):
            prompt = f"<{tag}>"
            idx = torch.tensor([tok.encode(prompt)], device=device)
            g = BracketGuard(tok.decode_token, tok.vocab_size) if guard else None
            end_id = tok.encode(f"</{tag}>")[0]
            stop = (lambda t, g=g: g.is_complete(t)) if g else (lambda t: t == end_id)
            out = generate(model, idx, max_new, temperature=0.8, top_p=0.95, processor=g, stop=stop,
                           generator=torch.Generator(device=device).manual_seed(i))
            text = tok.decode(out[0].tolist())[len(prompt):]
            text = text.split(f"</{tag}>")[0]
            try:
                json.loads(text)
                ok += 1
            except json.JSONDecodeError:
                pass
        res[tag] = ok / n
    res["_mean"] = round(sum(res.values()) / len(JSON_TAGS), 3)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/bedrock/ckpt.pt")
    ap.add_argument("--n", type=int, default=5, help="タグごとの生成数")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--skip_gen", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(a.ckpt, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    tok = load_tokenizer(ckpt["data"])

    val_text = (Path(ckpt["data"]) / "corpus" / "val.txt").read_text(encoding="utf-8")
    tags = per_tag_loss(model, tok, val_text, device)
    total = sum(v["tokens"] for v in tags.values())
    print("== タグ別 val loss ==")
    for t, v in tags.items():
        print(f"{t:15s} loss {v['loss']:.3f}  ppl {math.exp(v['loss']):8.1f}  tokens {v['tokens']:7,} ({v['tokens'] / total:5.1%})")
    if a.skip_gen:
        return
    for guard in (False, True):
        print(f"== JSON構文妥当性率 (json_guard={guard}) ==")
        print(json.dumps(json_validity(model, tok, device, a.n, guard, a.max_new_tokens), ensure_ascii=False))


if __name__ == "__main__":
    main()
