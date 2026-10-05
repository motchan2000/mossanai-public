"""STEP8: 学習済みチェックポイントから文章を生成する。

例:
    uv run python sample.py --ckpt out/shakespeare_char/ckpt.pt --prompt "ROMEO:" --top_p 0.9
    uv run python sample.py --ckpt out/bedrock/ckpt.pt --prompt "<BP_ENTITY>{" --json_guard
"""
from __future__ import annotations

import argparse

import torch

from model import GPT, GPTConfig
from sampling import BracketGuard, generate
from tokenizer_utils import load_tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8, help="0でGreedy")
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--json_guard", action="store_true", help="括弧対応を壊すトークンを禁止し、JSONが閉じたら停止")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(a.ckpt, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    tok = load_tokenizer(ckpt["data"])

    gen = torch.Generator(device=device).manual_seed(a.seed) if a.seed is not None else None
    prompt_ids = tok.encode(a.prompt)
    for i in range(a.num_samples):
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        guard = BracketGuard(tok.decode_token, tok.vocab_size, prefix=a.prompt) if a.json_guard else None
        out = generate(
            model, idx, a.max_new_tokens, a.temperature, a.top_k, a.top_p,
            processor=guard, stop=guard.is_complete if guard else None, generator=gen,
        )
        print(f"--- sample {i + 1} ---")
        print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
