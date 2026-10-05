"""STEP7 / STEP9: 学習ループ。

使い方:
    uv run python train.py --data data/shakespeare_char --out out/shakespeare_char
データディレクトリには train.bin / val.bin（uint16のトークン列）と meta.json が必要。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    # モデル
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    # 学習
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1, help="実効batch = batch_size × grad_accum")
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # 評価・ログ
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


class Batcher:
    def __init__(self, data_dir: Path, block_size: int, batch_size: int, device: str):
        self.data = {s: np.memmap(data_dir / f"{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}
        self.block_size, self.batch_size, self.device = block_size, batch_size, device

    def get(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.data[split]
        ix = torch.randint(len(data) - self.block_size - 1, (self.batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if self.device == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


def lr_at(it: int, a: argparse.Namespace) -> float:
    """線形warmup → cosine減衰でmin_lrまで。"""
    if it < a.warmup_iters:
        return a.lr * (it + 1) / a.warmup_iters
    if it >= a.max_iters:
        return a.min_lr
    ratio = (it - a.warmup_iters) / (a.max_iters - a.warmup_iters)
    return a.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (a.lr - a.min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, batcher: Batcher, iters: int, ctx) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = batcher.get(split)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    a = parse_args()
    data_dir, out_dir = Path(a.data), Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if use_bf16 else nullcontext()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    ckpt_path = out_dir / "ckpt.pt"
    start_iter, best_val = 0, float("inf")

    if a.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = GPTConfig(**ckpt["config"])
        model = GPT(cfg).to(device)
        model.load_state_dict(ckpt["model"])
        start_iter, best_val = ckpt["iter"] + 1, ckpt["best_val"]
    else:
        cfg = GPTConfig(
            vocab_size=meta["vocab_size"], block_size=a.block_size, n_layer=a.n_layer,
            n_head=a.n_head, d_model=a.d_model, dropout=a.dropout,
        )
        model = GPT(cfg).to(device)
    print(f"device={device} bf16={use_bf16} params={model.num_params() / 1e6:.2f}M")

    opt = model.configure_optimizer(a.lr, a.weight_decay, (0.9, 0.95), device)
    if a.resume and ckpt_path.exists():
        opt.load_state_dict(ckpt["optimizer"])

    batcher = Batcher(data_dir, cfg.block_size, a.batch_size, device)
    tokens_per_iter = a.batch_size * a.grad_accum * cfg.block_size
    print(f"tokens/iter={tokens_per_iter:,}")

    x, y = batcher.get("train")
    t0 = time.time()
    for it in range(start_iter, a.max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, a)

        if it % a.eval_interval == 0 or it == a.max_iters:
            losses = estimate_loss(model, batcher, a.eval_iters, ctx)
            print(f"[eval] iter {it}: train {losses['train']:.4f} / val {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {"model": model.state_dict(), "optimizer": opt.state_dict(), "config": cfg.to_dict(),
                     "iter": it, "best_val": best_val, "data": str(data_dir)},
                    ckpt_path,
                )
                print(f"  saved {ckpt_path} (best val {best_val:.4f})")
            if it == a.max_iters:
                break

        for _ in range(a.grad_accum):
            with ctx:
                _, loss = model(x, y)
                loss = loss / a.grad_accum
            x, y = batcher.get("train")  # GPU計算中に次のバッチを準備
            loss.backward()
        if a.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if it % a.log_interval == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            t0 = time.time()
            n = a.log_interval if it > start_iter else 1
            print(f"iter {it}: loss {loss.item() * a.grad_accum:.4f}, lr {lr_at(it, a):.2e}, "
                  f"{dt / n * 1000:.0f}ms/iter, {tokens_per_iter * n / dt:,.0f} tok/s")


if __name__ == "__main__":
    main()
