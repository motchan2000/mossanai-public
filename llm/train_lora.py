"""LoRAによる追加学習（自分専用モデルの学習）。

データ: {"prompt": ..., "completion": ...} のjsonl（複数可）。損失は completion 部分のみ。

    uv run python -m llm.train_lora --data data/sft/fim_train.jsonl --val data/sft/fim_val.jsonl --name fim_v1

7B（Instruct）は 4bit で読み込む QLoRA（8GB の GPU 用）。チャット用トークンは学習済みなので埋め込みは学習しない:

    uv run python -m llm.train_lora --base models/base/Qwen2.5-Coder-7B-Instruct --qlora --train_tokens \\
        --data ... --name v4 --tokens_per_batch 2048 --grad_accum 8
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint

from llm.fim import DEFAULT_BASE, PROJECT_ROOT, fast_attention

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(DEFAULT_BASE))
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--val", nargs="*", default=[])
    p.add_argument("--name", required=True)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--tokens_per_batch", type=int, default=4096, help="1マイクロバッチの最大トークン数（パディング込み）")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--ce_chunk", type=int, default=1024, help="損失を計算する位置の小分けの大きさ（7B は 256 でメモリのピークを下げる）")
    p.add_argument("--train_tokens", nargs="*", default=["<|im_start|>", "<|im_end|>"],
                   help="埋め込みも学習するトークン。ベースモデルではチャット用トークンが未学習で、"
                        "LoRAだけでは回答の終わりに <|im_end|> を出せない（v1で判明）")
    p.add_argument("--max_examples", type=int, default=0,
                   help="1マイクロバッチの最大件数（0=制限なし）。1 にするとパディングが無くなり、注意機構にマスクを渡さずに済む"
                        "（マスク付きは遅い実装になりメモリも増えるため、7B では 1 を推奨）")
    p.add_argument("--qlora", action="store_true", help="ベースを 4bit（NF4）で読み込む（7B を 8GB で学習するため）")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_examples(paths: list[str], tok, max_len: int) -> list[tuple[list[int], int]]:
    """(token ids, prompt長) のリスト。max_lenを超えるものは捨てる（FIMの構造を壊さないため切り詰めない）。"""
    out, dropped = [], 0
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            p = tok(r["prompt"], add_special_tokens=False).input_ids
            c = tok(r["completion"], add_special_tokens=False).input_ids
            if len(p) + len(c) > max_len:
                dropped += 1
                continue
            out.append((p + c, len(p)))
    print(f"loaded {len(out):,} examples from {paths} (dropped {dropped} over {max_len} tokens)")
    return out


def make_batches(examples, tokens_per_batch: int, rng: random.Random, max_examples: int = 0) -> list[list[int]]:
    """長さが近いものをまとめてパディングを減らす（バケット化してからシャッフル）。"""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]) + rng.random())
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(examples[i][0])
        if cur and (max(cur_max, L) * (len(cur) + 1) > tokens_per_batch or len(cur) == max_examples):
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def _chunk_ce(lm_head, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(lm_head(h).float(), y, reduction="sum")


def completion_loss(model, ids, attn, labels, chunk: int = 1024) -> tuple[torch.Tensor, int]:
    """completion部分だけにLM Headを適用して損失（合計, トークン数）を返す。

    model(labels=...) は全位置×語彙15万のlogitsをfloat32で作るため、6kトークンで約3.7GBになり
    8GBのGPUではメモリが溢れて極端に遅くなる。必要な位置だけを、チェックポイント付きの小分けで計算する。
    """
    inner = model.get_base_model()  # LoRA層は注入済みなのでこのforwardでも有効
    # パディングが無ければマスクを渡さない（マスク付きだとSDPAが T×T 行列を作る遅い実装になり、長系列でOOMする）
    mask_arg = None if bool(attn.all()) else attn
    # 注意: 呼び出し側で fast_attention() の中で forward と backward の両方を行うこと
    # （チェックポイントの再計算で別カーネルが選ばれると保存テンソル数が食い違いエラーになる）
    hidden = inner.model(input_ids=ids, attention_mask=mask_arg).last_hidden_state
    tgt = labels[:, 1:]
    mask = tgt != -100
    h = hidden[:, :-1][mask]
    y = tgt[mask]
    # lm_head はモジュールとして呼ぶ（trainable_token_indices の差分がここで加わるため、重みの直接参照は不可）
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.size(0), chunk):
        total = total + checkpoint(_chunk_ce, inner.lm_head, h[i:i + chunk], y[i:i + chunk], use_reentrant=False)
    return total, int(y.numel())


def collate(examples, idx: list[int], pad_id: int, device: str):
    L = max(len(examples[i][0]) for i in idx)
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    labels = torch.full((len(idx), L), -100, dtype=torch.long)
    attn = torch.zeros((len(idx), L), dtype=torch.long)
    for row, i in enumerate(idx):
        toks, plen = examples[i]
        ids[row, : len(toks)] = torch.tensor(toks)
        labels[row, plen : len(toks)] = torch.tensor(toks[plen:])
        attn[row, : len(toks)] = 1
    return ids.to(device), labels.to(device), attn.to(device)


@torch.no_grad()
def evaluate(model, examples, batches, pad_id) -> float:
    model.eval()
    total, n = 0.0, 0
    for idx in batches:
        ids, labels, attn = collate(examples, idx, pad_id, "cuda")
        with fast_attention(), torch.autocast("cuda", dtype=torch.bfloat16):
            loss_sum, k = completion_loss(model, ids, attn, labels)
        total += loss_sum.item()
        n += k
    model.train()
    return total / max(n, 1)


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    a = parse_args()
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    out_dir = PROJECT_ROOT / "out" / "lora" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    pad_id = tok.convert_tokens_to_ids("<|endoftext|>")
    train = load_examples(a.data, tok, a.max_len)
    val = load_examples(a.val, tok, a.max_len) if a.val else []
    val_batches = make_batches(val, a.tokens_per_batch, random.Random(0), a.max_examples) if val else []

    if a.qlora:
        from transformers import BitsAndBytesConfig

        # 注意: peft の prepare_model_for_kbit_training は使わない（量子化されない埋め込み・lm_head を float32 に
        # 変換するため、7B では語彙15万×3584 の2つで約4.4GBになり 8GB に収まらない）。bf16 のまま学習する
        # 入力の埋め込み（7B で 1.1GB）は CPU に置く。表を引くだけなので遅くならず、4096 トークンまで 8GB に収まる
        # （GPU のままだと 4096 で VRAM を超え、Windows が共有メモリに逃がして 3 分の 1 の速さになる）
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True)
        device_map = {"model.embed_tokens": "cpu", "model.layers": 0, "model.norm": 0, "model.rotary_emb": 0,
                      "lm_head": 0}
        model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map=device_map,
                                                     quantization_config=bnb)
    else:
        model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    token_ids = [tok.convert_tokens_to_ids(t) for t in a.train_tokens]
    extra = {"trainable_token_indices": {"embed_tokens": token_ids}} if token_ids else {}
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM", **extra))
    model.print_trainable_parameters()

    epoch_batches = make_batches(train, a.tokens_per_batch, rng, a.max_examples)
    total_micro = int(len(epoch_batches) * a.epochs)
    total_steps = math.ceil(total_micro / a.grad_accum)
    params = [p for p in model.parameters() if p.requires_grad]
    if a.qlora:
        import bitsandbytes as bnb_optim

        # 8bit の AdamW（状態が 1/4 になり、7B の LoRA で約 0.3GB 節約）
        opt = bnb_optim.optim.AdamW8bit(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    else:
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))

    def lr_at(step: int) -> float:
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        r = (step - a.warmup) / max(1, total_steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(r, 1.0))))

    (out_dir / "train_args.json").write_text(json.dumps(vars(a), indent=2, ensure_ascii=False), encoding="utf-8")
    log = open(out_dir / "log.txt", "a", encoding="utf-8")

    def emit(msg: str) -> None:
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    emit(f"steps={total_steps} micro_batches={total_micro} grad_accum={a.grad_accum}")
    if val_batches:
        emit(f"[eval] step 0: val {evaluate(model, val, val_batches, pad_id):.4f}")

    model.train()
    batches = list(epoch_batches)
    step, micro, t0, tokens, run_loss = 0, 0, time.time(), 0, 0.0
    best = float("inf")
    while micro < total_micro:
        if not batches:
            batches = make_batches(train, a.tokens_per_batch, rng, a.max_examples)
        ids, labels, attn = collate(train, batches.pop(), pad_id, "cuda")
        with fast_attention():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss_sum, k = completion_loss(model, ids, attn, labels, chunk=a.ce_chunk)
                loss = loss_sum / max(k, 1) / a.grad_accum
            loss.backward()
        run_loss += loss.item()
        tokens += int(attn.sum())
        micro += 1
        if micro % a.grad_accum:
            continue
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % a.log_every == 0:
            dt = time.time() - t0
            emit(f"step {step}/{total_steps}: loss {run_loss / a.log_every:.4f} lr {lr_at(step):.2e} "
                 f"{tokens / dt:,.0f} tok/s mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
            t0, tokens, run_loss = time.time(), 0, 0.0
        if val_batches and (step % a.eval_every == 0 or step == total_steps):
            v = evaluate(model, val, val_batches, pad_id)
            emit(f"[eval] step {step}: val {v:.4f}")
            if v < best:
                best = v
                model.save_pretrained(out_dir / "best")
                emit(f"  saved best ({v:.4f})")
    model.save_pretrained(out_dir / "last")
    emit("done")


if __name__ == "__main__":
    main()
