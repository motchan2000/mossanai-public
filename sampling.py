"""STEP8: 推論ロジック。

- Greedy / Top-k / Top-p(nucleus) サンプリング
- JSON括弧の対応を壊すトークンを生成時に禁止する制約（BracketGuard）
  将来的には outlines / llguidance による JSON Schema 制約付き生成に差し替える前提で、
  「logitsを受け取ってマスクする」インターフェイス（LogitsProcessor）に揃えている。
"""
from __future__ import annotations

from typing import Callable, Protocol

import torch
import torch.nn.functional as F


class LogitsProcessor(Protocol):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor: ...
    def update(self, token_id: int) -> None: ...


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits: [B, vocab] → 次トークンID [B, 1]。temperature<=0 ならGreedy。"""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # 累積確率がtop_pを超えた「次」以降を捨てる（最上位トークンは必ず残る）
        remove = cum - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    processor: LogitsProcessor | None = None,
    stop: Callable[[int], bool] | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """idx: [1, T] のプロンプトに続けて生成する（バッチ1前提。制約生成の状態管理を単純にするため）。"""
    model.eval()
    block_size = model.cfg.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :].float()
        if processor is not None:
            logits = processor(logits)
        next_id = sample_next(logits, temperature, top_k, top_p, generator)
        idx = torch.cat([idx, next_id], dim=1)
        tid = int(next_id.item())
        if processor is not None:
            processor.update(tid)
        if stop is not None and stop(tid):
            break
    return idx


# ---------------------------------------------------------------------------
# JSON括弧の簡易バリデータ
# ---------------------------------------------------------------------------
_PAIRS = {"}": "{", "]": "["}


class JsonScanState:
    """文字列リテラル内を無視しつつ {} [] の対応をスタックで追跡する。"""

    __slots__ = ("stack", "in_string", "escape", "opened")

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.in_string = False
        self.escape = False
        self.opened = False  # 一度でも括弧を開いたか

    def copy(self) -> "JsonScanState":
        s = JsonScanState()
        s.stack = self.stack.copy()
        s.in_string, s.escape, s.opened = self.in_string, self.escape, self.opened
        return s

    def feed(self, text: str) -> bool:
        """textを読み進める。対応の崩れ（開いていない閉じ括弧・種類違い）があればFalse。"""
        for ch in text:
            if self.in_string:
                if self.escape:
                    self.escape = False
                elif ch == "\\":
                    self.escape = True
                elif ch == '"':
                    self.in_string = False
                continue
            if ch == '"':
                self.in_string = True
            elif ch in "{[":
                self.stack.append(ch)
                self.opened = True
            elif ch in "}]":
                if not self.stack or self.stack[-1] != _PAIRS[ch]:
                    return False
                self.stack.pop()
        return True

    @property
    def complete(self) -> bool:
        """最上位のJSON値が閉じ切った状態か。"""
        return self.opened and not self.stack and not self.in_string


def check_brackets(text: str) -> tuple[bool, JsonScanState]:
    state = JsonScanState()
    ok = state.feed(text)
    return ok, state


class BracketGuard:
    """括弧対応を壊すトークンのlogitsを -inf にするLogitsProcessor。

    毎ステップ全語彙を走査すると遅いので、括弧/引用符/バックスラッシュを含む
    トークンだけを事前に抽出しておき、それらだけ検査する。
    """

    def __init__(self, decode_token: Callable[[int], str], vocab_size: int, prefix: str = ""):
        self.decode_token = decode_token
        self.state = JsonScanState()
        self.state.feed(prefix)
        texts = {i: decode_token(i) for i in range(vocab_size)}
        self.special = {i: t for i, t in texts.items() if any(c in t for c in '{}[]"\\')}

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        banned = [i for i, t in self.special.items() if not self.state.copy().feed(t)]
        if banned:
            logits = logits.clone()
            logits[:, banned] = float("-inf")
        return logits

    def update(self, token_id: int) -> None:
        self.state.feed(self.decode_token(token_id))

    def is_complete(self, _token_id: int) -> bool:
        """stop引数に渡すと、JSONが閉じた時点で生成を止められる。"""
        return self.state.complete
