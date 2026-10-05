"""設計図 STEP3〜6 の完了基準をテスト化したもの。"""
import math

import torch

from model import GPT, GPTConfig, apply_rope, build_rope_cache

torch.manual_seed(0)


def small_cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=97, block_size=32, n_layer=2, n_head=4, d_model=64)
    base.update(kw)
    return GPTConfig(**base)


def test_embedding_shape():
    """STEP3: [batch, seq_len] → [batch, seq_len, d_model]"""
    model = GPT(small_cfg())
    idx = torch.randint(0, 97, (3, 20))
    assert model.tok_emb(idx).shape == (3, 20, 64)


def test_rope_depends_only_on_relative_position():
    """STEP3: RoPE適用後の q·k は絶対位置ではなく相対距離だけで決まる。"""
    head_dim = 16
    cos, sin = build_rope_cache(head_dim, 64, 10000.0)
    q, k = torch.randn(head_dim), torch.randn(head_dim)

    def score(m: int, n: int) -> float:
        qm = apply_rope(q.view(1, 1, 1, -1), cos[m], sin[m]).flatten()
        kn = apply_rope(k.view(1, 1, 1, -1), cos[n], sin[n]).flatten()
        return (qm @ kn).item()

    assert math.isclose(score(5, 2), score(40, 37), rel_tol=1e-4, abs_tol=1e-5)
    assert not math.isclose(score(5, 2), score(5, 4), rel_tol=1e-3)


def test_causal_mask_blocks_future_tokens():
    """STEP4: 位置tより後のトークンを変えても、位置t以前のlogitsは変わらない。"""
    model = GPT(small_cfg()).eval()
    idx = torch.randint(0, 97, (1, 16))
    changed = idx.clone()
    changed[0, 10:] = (changed[0, 10:] + 1) % 97
    with torch.no_grad():
        a, _ = model(idx, targets=idx)  # targetsを渡すと全位置のlogitsが返る
        b, _ = model(changed, targets=changed)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5)
    assert not torch.allclose(a[0, 10:], b[0, 10:], atol=1e-5)


def test_ten_layers_forward_backward_stable():
    """STEP5: 10層重ねても勾配が爆発/消失しない。"""
    model = GPT(small_cfg(n_layer=10))
    idx = torch.randint(0, 97, (4, 32))
    _, loss = model(idx, targets=idx)
    loss.backward()
    first = model.blocks[0].attn.qkv.weight.grad.norm().item()
    last = model.blocks[-1].attn.qkv.weight.grad.norm().item()
    assert math.isfinite(first) and math.isfinite(last)
    assert first > 1e-6 and last > 1e-6
    assert 0.01 < first / last < 100  # 層間で勾配の大きさが極端に違わない


def test_initial_loss_is_near_uniform():
    """STEP6: ランダム初期化で forward が通り、lossがほぼ ln(vocab) になる。"""
    model = GPT(small_cfg())
    idx = torch.randint(0, 97, (8, 32))
    targets = torch.randint(0, 97, (8, 32))  # idxと同じにすると重み共有の影響で「入力と同じトークン」が有利になる
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (8, 32, 97)
    assert abs(loss.item() - math.log(97)) < 0.3


def test_weight_tying():
    """STEP6: EmbeddingとLM Headが同じパラメータを共有している。"""
    model = GPT(small_cfg())
    assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
    untied = sum(p.numel() for p in model.parameters()) + model.lm_head.weight.numel()
    assert model.num_params() == untied - 97 * 64


def test_inference_returns_last_position_only():
    model = GPT(small_cfg()).eval()
    logits, loss = model(torch.randint(0, 97, (2, 10)))
    assert logits.shape == (2, 1, 97) and loss is None


def test_can_overfit_a_tiny_batch():
    """学習ループの基本動作: 同じバッチを繰り返せばlossが大きく下がる。"""
    model = GPT(small_cfg())
    opt = model.configure_optimizer(3e-3, 0.0, (0.9, 0.95), "cpu")
    idx = torch.randint(0, 97, (2, 32))
    x, y = idx[:, :-1], idx[:, 1:]
    _, first = model(x, y)
    for _ in range(100):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first.item() * 0.2
