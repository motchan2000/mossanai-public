"""STEP8: サンプリングとJSON括弧制約のテスト。"""
import torch

from model import GPT, GPTConfig
from sampling import BracketGuard, check_brackets, generate, sample_next


def test_greedy_picks_argmax():
    logits = torch.tensor([[0.1, 3.0, 0.5]])
    assert sample_next(logits, temperature=0).item() == 1


def test_top_k_only_samples_from_top_k():
    logits = torch.tensor([[5.0, 4.0, 0.0, 0.0, 0.0]])
    g = torch.Generator().manual_seed(0)
    picks = {sample_next(logits, 1.0, top_k=2, generator=g).item() for _ in range(200)}
    assert picks <= {0, 1}


def test_top_p_keeps_top_token_and_cuts_tail():
    # 確率おおよそ [0.7, 0.2, 0.1 を残り4つで分割]
    probs = torch.tensor([[0.70, 0.20, 0.025, 0.025, 0.025, 0.025]])
    g = torch.Generator().manual_seed(0)
    picks = {sample_next(probs.log(), 1.0, top_p=0.8, generator=g).item() for _ in range(300)}
    assert picks == {0, 1}
    # top_pが極小でも最上位トークンは必ず残る
    assert sample_next(probs.log(), 1.0, top_p=1e-6).item() == 0


def test_check_brackets_ignores_strings():
    ok, st = check_brackets('{"a": "}]{[", "b": [1, {"c": 2}]}')
    assert ok and st.complete
    ok, _ = check_brackets('{"a": [1, 2}')
    assert not ok
    ok, st = check_brackets('{"a": "x\\"}')  # エスケープされた引用符で文字列はまだ閉じていない
    assert ok and st.in_string


class ToyVocab:
    tokens = ["{", "}", "[", "]", '"k"', ":", "1", ",", "}}", "]}"]

    def decode_token(self, i: int) -> str:
        return self.tokens[i]


def test_bracket_guard_bans_invalid_closers():
    v = ToyVocab()
    guard = BracketGuard(v.decode_token, len(v.tokens), prefix='{"k": [')
    masked = guard(torch.zeros(1, len(v.tokens)))
    banned = {i for i in range(len(v.tokens)) if masked[0, i] == float("-inf")}
    # スタックは "{[" なので "}" "}}" は不正、"]" "]}" は正当
    assert banned == {v.tokens.index("}"), v.tokens.index("}}")}


def test_generate_with_guard_produces_balanced_json():
    """未学習モデルでも、制約付き生成なら括弧対応は絶対に崩れない。"""
    v = ToyVocab()
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=len(v.tokens), block_size=64, n_layer=1, n_head=2, d_model=16))
    for seed in range(20):
        guard = BracketGuard(v.decode_token, len(v.tokens))
        idx = torch.tensor([[0]])  # "{"
        guard.update(0)
        out = generate(model, idx, 40, temperature=1.5, processor=guard,
                       stop=guard.is_complete, generator=torch.Generator().manual_seed(seed))
        text = "".join(v.decode_token(i) for i in out[0].tolist())
        ok, _ = check_brackets(text)
        assert ok, text
