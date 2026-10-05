"""学習済みLoRAをベースモデルに統合し、独自モデルとして Ollama に登録する。

    uv run python -m llm.export --adapter out/lora/v1/best --name <モデル名> [--outtype q8_0] [--create]

手順:
  1. LoRA を統合して models/merged/<name>/ に safetensors で保存
  2. llama.cpp の convert_hf_to_gguf.py で GGUF に変換（同時に量子化。tools/llama.cpp, sources.json 参照）
     ※ Ollama 0.34 の safetensors 直接取り込み（MLX経路）は Qwen2 非対応のため GGUF 経由にする
  3. Modelfile を作成（FIM補完=VSCode用 と ChatML=CLI用 の両方に対応するテンプレート）
  4. --create 指定時は `ollama create <name>` を実行

7B は q8_0（約8GB）だと 8GB の GPU に載り切らないので --quant q4_k_m（約4.7GB）を使う。
convert_hf_to_gguf.py は 4bit の K-quant を作れないため、bf16 の GGUF を作ってから llama.cpp の
llama-quantize（tools/llama-bin/llama-quantize.exe）で量子化する。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from llm.chat import SYSTEM
from llm.fim import DEFAULT_BASE, PROJECT_ROOT

# FIM: Continue 等は prompt/suffix を渡してくる → <|fim_prefix|>…<|fim_middle|>
# チャット: 学習時と同じ ChatML（system → user → assistant）
TEMPLATE = r'''{{- if .Suffix }}<|fim_prefix|>{{ .Prompt }}<|fim_suffix|>{{ .Suffix }}<|fim_middle|>
{{- else if .Messages }}<|im_start|>system
{{ if .System }}{{ .System }}{{ else }}__SYSTEM__{{ end }}<|im_end|>
{{ range .Messages }}{{ if ne .Role "system" }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}{{ end }}<|im_start|>assistant
{{ else }}{{ .Prompt }}{{ end }}'''

MODELFILE = """FROM {src}
TEMPLATE \"\"\"{template}\"\"\"
PARAMETER stop <|im_end|>
PARAMETER stop <|endoftext|>
PARAMETER stop <|fim_pad|>
PARAMETER stop <|file_sep|>
PARAMETER temperature 0.2
PARAMETER num_ctx 8192
LICENSE \"\"\"{license}\"\"\"
"""


def merge(base: Path, adapter: Path, out: Path) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    for f in ("LICENSE",):
        if (base / f).exists():
            shutil.copy(base / f, out / f)
    (out / "NOTICE.md").write_text(
        f"このモデルは Qwen/{base.name}（Apache License 2.0）をベースに、\n"
        f"Minecraft Bedrock アドオン開発向けに追加学習（LoRA: {adapter.as_posix()}）したものです。\n",
        encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--name", required=True, help="Ollamaに登録するモデル名（英小文字・数字・-）")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--outtype", default="q8_0", help="GGUFの型: q8_0（推奨, 1.5GB）/ bf16 / f16")
    ap.add_argument("--quant", default=None, help="llama-quantize で量子化する型（例: q4_k_m）。指定時 --outtype は bf16 にする")
    ap.add_argument("--create", action="store_true", help="ollama create まで実行する")
    ap.add_argument("--skip_merge", action="store_true", help="統合済みなら再統合しない")
    a = ap.parse_args()

    out = PROJECT_ROOT / "models" / "merged" / a.name
    if not (a.skip_merge and (out / "config.json").exists()):
        merge(a.base, a.adapter, out)
    gguf = out / f"{a.name}-{a.outtype}.gguf"
    convert = PROJECT_ROOT / "tools" / "llama.cpp" / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise SystemExit("tools/llama.cpp が見つかりません。README の「GGUF変換ツール」の手順で取得してください")
    if a.quant:
        a.outtype = "bf16"
        gguf = out / f"{a.name}-bf16.gguf"
    subprocess.run([sys.executable, str(convert), str(out), "--outfile", str(gguf), "--outtype", a.outtype],
                   check=True)
    if a.quant:
        quantize = PROJECT_ROOT / "tools" / "llama-bin" / "llama-quantize.exe"
        if not quantize.exists():
            raise SystemExit(f"{quantize} がありません（llama.cpp の Windows 用リリースから取得してください）")
        q = out / f"{a.name}-{a.quant}.gguf"
        subprocess.run([str(quantize), str(gguf), str(q), a.quant.upper()], check=True)
        gguf.unlink()  # bf16 の中間ファイル（7B で約15GB）は消す
        gguf = q
    license_text = (a.base / "LICENSE").read_text(encoding="utf-8") if (a.base / "LICENSE").exists() else ""
    modelfile = out / "Modelfile"
    # newline="\n" 必須: Windowsの既定（\r\n変換）だとテンプレートに\rが混ざり、
    # 学習時と異なるプロンプトになって出力が崩れる
    modelfile.write_text(MODELFILE.format(src=f"./{gguf.name}", template=TEMPLATE.replace("__SYSTEM__", SYSTEM),
                                          license=license_text.replace('"""', "'''")),
                         encoding="utf-8", newline="\n")
    print(f"GGUF: {gguf}")
    if a.create:
        cmd = ["ollama", "create", a.name, "-f", str(modelfile)]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=out)


if __name__ == "__main__":
    main()
