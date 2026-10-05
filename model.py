"""GPT型言語モデル（STEP3〜6）。

構成: Token Embedding → [Pre-LN Transformer Block] × N → LayerNorm → LM Head（重み共有）
位置情報はRoPE（学習パラメータなし）で注入する。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256  # 最大context長
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.0
    bias: bool = False  # Linear/LayerNormにbiasを付けるか（無しの方が少し軽く、性能差はほぼない）
    rope_theta: float = 10000.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# STEP3: Positional Embedding（RoPE）
# ---------------------------------------------------------------------------
def build_rope_cache(head_dim: int, max_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE用のcos/sinテーブル [max_len, head_dim] を作る。"""
    assert head_dim % 2 == 0, "RoPEにはhead_dimが偶数である必要があります"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)  # [T, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, n_head, T, head_dim] に回転位置埋め込みを適用する。"""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


# ---------------------------------------------------------------------------
# STEP4: Causal Self-Attention
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.proj.is_residual_proj = True
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] → [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # PyTorch 2.x: Flash Attention等の高速カーネルを自動選択。is_causalで未来を遮断。
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(out))


# ---------------------------------------------------------------------------
# STEP5: Transformer Block（Pre-LN）
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.proj.is_residual_proj = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# STEP6: GPT本体
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # 重み共有: Embedding と LM Head で同じ行列を使う（vocab×d_model分のパラメータ削減）
        self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(cfg.d_model // cfg.n_head, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "is_residual_proj", False):
                # 残差経路の出力は層数に応じて縮小（GPT-2方式）→ 深くしても発散しにくい
                std /= math.sqrt(2 * self.cfg.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # 重み共有しているのでparameters()は同じテンソルを1回しか数えない
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"系列長 {T} が block_size {self.cfg.block_size} を超えています"
        cos = self.rope_cos[:T].to(idx.device)
        sin = self.rope_sin[:T].to(idx.device)

        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        else:
            # 推論時は最後の位置だけ計算すれば十分
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizer(
        self, lr: float, weight_decay: float, betas: tuple[float, float], device_type: str
    ) -> torch.optim.AdamW:
        # 2次元以上（行列・Embedding）だけweight decay、bias/LayerNormは対象外
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=(device_type == "cuda"))
