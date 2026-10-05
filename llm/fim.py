"""Qwen2.5-Coder の FIM（Fill-in-the-Middle）プロンプトと生成。"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = PROJECT_ROOT / "models" / "base" / "Qwen2.5-Coder-1.5B"

FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"
FILE_SEP = "<|file_sep|>"


def fast_attention():
    """SDPAのカーネル優先順位を指定する。

    Windows版PyTorchはFlash Attention非搭載で、transformersがGQA(enable_gqa)で呼ぶと
    Memory-Efficientカーネルも使えず、T×Tの重み行列を作るmath実装に落ちる（長系列でOOM）。
    cuDNNカーネルはGQAに対応しているので最優先にする。
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
                       set_priority=True)


def fim_prompt(prefix: str, suffix: str, path: str | None = None) -> str:
    """Qwen2.5-Coder形式のFIMプロンプト。pathを付けるとファイル種別のヒントになる。"""
    head = f"{FILE_SEP}{path}\n" if path else ""
    return f"{FIM_PREFIX}{head}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


def load(model_path: str | Path = DEFAULT_BASE, adapter: str | Path | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


@dataclass
class Completion:
    text: str
    seconds: float
    new_tokens: int


@torch.no_grad()
def complete(tok, model, prompt: str, max_new_tokens: int, stop_at_newline: bool = False,
             max_prompt_tokens: int = 4096) -> Completion:
    ids = tok(prompt, return_tensors="pt").input_ids
    if ids.size(1) > max_prompt_tokens:
        raise ValueError(f"prompt too long: {ids.size(1)} tokens")
    ids = ids.to(model.device)
    stop_ids = [tok.convert_tokens_to_ids(t) for t in ("<|endoftext|>", "<|fim_pad|>", "<|file_sep|>",
                                                       "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")]
    kwargs = {}
    if stop_at_newline:
        kwargs["stop_strings"] = ["\n"]
        kwargs["tokenizer"] = tok
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=stop_ids,
                         pad_token_id=stop_ids[0], **kwargs)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    new = out[0, ids.size(1):]
    text = tok.decode(new, skip_special_tokens=True)
    # <|fim_pad|> 等は「特殊トークン」扱いになっておらず skip されないので、最初に現れた位置で切る
    for marker in ("<|fim_pad|>", "<|endoftext|>", "<|file_sep|>", FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE):
        text = text.split(marker)[0]
    if stop_at_newline:
        text = text.split("\n")[0]
    return Completion(text, dt, int(new.numel()))
