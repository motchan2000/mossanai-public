# mossanai — Minecraft Bedrock アドオン開発特化のローカルLLM

Qwen2.5-Coder-1.5B をベースに、統合版（Bedrock Edition）アドオン開発向けに追加学習した自分専用モデルと、
その学習・評価・検証ツール一式。設計図: [docs/design.md](docs/design.md)

> **このリポジトリについて**: 学習・評価・検証ツールのコードの公開版です。
> 学習データ（Mojang の bedrock-samples 由来のものや作者の自作アドオン由来のもの）と学習済みの重みは含みません。
> データは `sources.json` の版のサンプルとツールで各自作り直せます。

## セットアップ
必要なもの: Git、[uv](https://docs.astral.sh/uv/)、NVIDIA GPU（学習する場合。RTX 50系はCUDA 12.8以上）、[Ollama](https://ollama.com/)（モデルを使う場合）、Node.js（Script API の型チェック）
```bash
git clone https://github.com/motchan2000/mossanai-public.git
cd mossanai-public
uv sync                                          # Python 3.12 環境と依存関係
uv run python -m llm.fetch_base                  # ベースモデル（sources.json の版, 約3GB）
cd data/bedrock && uv run python fetch_samples.py && cd ../..   # Mojang bedrock-samples（検証ツールに必要。Minecraft EULA に同意のうえ取得）
cd script_check && npm install && cd ..          # Script API の型チェック（@minecraft/server の型定義）
uv run pytest                                    # 動作確認
```
学習データは `llm/build_fim_data.py`・`build_repair_data.py`・`build_gen_data.py`（bedrock-samples から生成）で作れます。
自作アドオン由来の教材（`llm/user_code.py`・`backtranslate.py`・`distill.py`）は、各自のアドオンのフォルダから作る想定です。

## 環境
- Python 3.12（uv管理）/ PyTorch 2.11 + CUDA 12.8（RTX 50系はcu128以上が必須）
- `uv sync` で依存関係をインストール

## ファイル構成
| ファイル | 内容 | 設計図 |
|---|---|---|
| `model.py` | GPT本体（RoPE, SDPA Attention, Pre-LN, 重み共有） | STEP3〜6 |
| `sampling.py` | Greedy / Top-k / Top-p、JSON括弧制約（BracketGuard） | STEP8 |
| `train.py` | 学習ループ（AdamW, warmup+cosine, bf16, 勾配累積） | STEP7, 9 |
| `sample.py` | チェックポイントから生成 | STEP8 |
| `tokenizer_utils.py` | char / BPE トークナイザの共通窓口 | |
| `data/shakespeare_char/` | パイプライン検証用データ | STEP7 |
| `data/bedrock/fetch_samples.py` | Mojang公式 bedrock-samples をバージョン固定で取得 | STEP9 |
| `data/bedrock/script_api.py` | Script APIメタデータ → `.d.ts` 変換 | STEP9 |
| `data/bedrock/build_corpus.py` | 構造タグ付きコーパス生成 | STEP9 |
| `data/bedrock/train_tokenizer.py` | Bedrock特化BPEの学習・GPT-2比較 | STEP2 |
| `data/bedrock/prepare.py` | コーパス → train.bin / val.bin | STEP9 |

## アドオン開発ツール（bedrock_tools / LLM不要）
```bash
uv run python -m bedrock_tools new "My Addon" --description "説明" --script --out ./packs   # BP/RP雛形（UUID自動生成）
uv run python -m bedrock_tools check ./packs -v                                         # 検証（-vで警告も表示）
```
| モジュール | 内容 |
|---|---|
| `schema_validator.py` | Mojang公式JSON Schemaで検証（format_versionに応じてスキーマのバージョンを選択） |
| `semantic.py` | スキーマで拾えない意味チェック: 存在しないコンポーネント名（候補提示）、未定義component_groupへの参照、型定義なしコンポーネントのvanilla実績ベース型チェック |
| `manifest.py` | manifestの検証（UUID形式・重複、バージョン形式、script module、`@minecraft/server`の実在バージョン、BP/RP相互依存）と生成 |
| `check.py` | フォルダ全体の検証 |

公式スキーマの限界と対処:
- entityコンポーネント406個のうち142個は型定義なし（"Dynamic value"）→ vanillaの使用実績から型を推定して警告
- 未知のキーを禁止していない → コンポーネント名の存在チェックを `semantic.py` で実施
- `oneOf` の選択肢が重なっている → `anyOf` として扱う
- manifestスキーマは format_version 3 用 → format 2 は `manifest.py` で検証
- vanillaの behavior_pack 全体で **エラー0件** になることを確認済み（誤検出なし）

## 自分専用モデル（Qwen2.5-Coder-1.5B + LoRA）
ベース: [Qwen/Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B)（Apache 2.0, FIM対応）→ `models/base/`

| ファイル | 内容 |
|---|---|
| `llm/build_fim_data.py` | FIM（VSCode補完）学習データ: trainファイルに行途中/行/ブロック/範囲の穴を作る |
| `llm/build_repair_data.py` | 「検証エラー→修正」データ: 正しいファイルに誤りを入れ、検証ツールが検出したものだけ採用 |
| `llm/synth_entity.py` / `synth_item_block.py` / `build_gen_data.py` | 「日本語の指示→ファイル生成」合成データ（全件検証エラー0） |
| `llm/train_lora.py` | LoRA学習（bf16, completion部分のみ損失, 8GB対応） |
| `llm/eval_fim.py` / `eval_repair.py` / `eval_gen.py` | 評価（同じ問題でベース/学習後を比較） |
| `llm/export.py` | LoRA統合 → Ollamaに独自モデルとして登録 |
| `mcaddon/` | CLI: 指示 → 生成 → 検証 → 自己修正 → パックに書き込み |

```bash
uv run python -m llm.train_lora --data data/sft/fim_train.jsonl data/sft/repair_train.jsonl data/sft/gen_train.jsonl \
  --val data/sft/fim_val.jsonl data/sft/repair_val.jsonl data/sft/gen_val.jsonl --name v1 --max_len 6144 --tokens_per_batch 6144
uv run python -m llm.export --adapter out/lora/v1/best --name <モデル名> --create
uv run python -m mcaddon gen "夜だけ湧く、近づくと爆発する敵Mob。IDはmy:bomber" --model <モデル名> --pack packs/MyAddon
```

8GB GPU（Windows）で学習するための工夫:
- `model(labels=...)` は全位置×語彙15万のlogitsをfp32で作り6kトークンで約3.7GB → completion位置だけ小分けに計算
- Windows版PyTorchはFlash Attention非搭載 + GQAでmath実装に落ちてOOM → cuDNN SDPAを優先（`llm/fim.py: fast_attention`）

## Script API 対応（v4: Qwen2.5-Coder-7B-Instruct + QLoRA、CLI 生成用）
補完（VSCode）は速さ優先で 1.5B（v2）のまま、CLI の生成を 7B に切り替える計画。

| 手順 | ファイル | 内容 |
|---|---|---|
| 依頼文 | `llm/backtranslate.py` / `llm/request_templates.py` | ユーザーのコードから逆算した依頼（968件）+ 分野を補うテンプレート（225件） |
| 蒸留 | `llm/distill.py` | 先生（qwen3-coder:30b + RAG + お手本 + 型チェック/BDS 実行テストで自己修正）の合格答えだけを保存。再開可能・`--until HH:MM` で停止 |
| 教材 | `llm/build_v4_data.py` | 実際の利用時と同じプロンプト（`mcaddon/script_gen.py`）で gen / fix / JSON 維持の教材 |
| 学習 | `llm/train_lora.py --qlora` | 4bit（NF4）で読み込み、埋め込みは bf16 のまま（8GB に収めるため） |
| 書き出し | `llm/export.py --quant q4_k_m` | bf16 GGUF → `tools/llama-bin/llama-quantize.exe` で 4bit（約4.7GB） |
| 利用 | `mcaddon gen --model mossanai-v4 [--runtime]` | 生成 → 型チェック（+ BDS 実行テスト）→ 自己修正 → パックに書き込み、manifest の依存も追加 |

本番テスト問題（`data/eval/user_tasks.jsonl` の split=test）に似た依頼は、蒸留・教材の両方で除外している（文字バイグラムの類似度 0.35 以上）。

```bash
uv run python -m llm.distill --until 14:45     # GPU。先生モデルで教材づくり（1件 約1〜4分）
uv run python -m llm.build_v4_data             # CPU
uv run python -m llm.train_lora --base models/base/Qwen2.5-Coder-7B-Instruct --qlora --train_tokens --max_examples 1 --ce_chunk 256 \
  --data data/sft/v4_train.jsonl --val data/sft/v4_val.jsonl --name v4 --max_len 4096 --tokens_per_batch 4096 --grad_accum 8
uv run python -m llm.export --adapter out/lora/v4/best --name mossanai-v4 \
  --base models/base/Qwen2.5-Coder-7B-Instruct --quant q4_k_m --create
uv run python -m llm.eval_user --models mossanai-v4 --rag --compact --runtime
```

## バージョン管理
| 対象 | 管理 |
|---|---|
| コード・テスト・設定・`uv.lock` | Git（このリポジトリ） |
| ベースモデル・bedrock-samples・BDS | 管理外（`sources.json` の版で再取得） |
| 学習データ・学習途中のチェックポイント・統合モデル | 管理外（各自で生成） |

## 使い方
```bash
uv run pytest                                   # テスト

# STEP7: パイプライン検証
uv run python data/shakespeare_char/prepare.py
uv run python train.py --data data/shakespeare_char --out out/shakespeare_char
uv run python sample.py --ckpt out/shakespeare_char/ckpt.pt --prompt "ROMEO:" --top_p 0.9

# STEP2/9: Bedrockデータ
cd data/bedrock
uv run python fetch_samples.py
uv run python build_corpus.py
uv run python train_tokenizer.py
uv run python prepare.py
cd ../..
uv run python train.py --data data/bedrock --out out/bedrock --block_size 512 --batch_size 32 --dropout 0.1 --lr 6e-4
uv run python sample.py --ckpt out/bedrock/ckpt.pt --prompt "<BP_ENTITY>{" --json_guard
```

## 設計図からの変更点
- **トークナイザ**: GPT-2方式の事前分割では `"minecraft:entity"` が原理的に1トークンにならないため、
  JSON文字列リテラルと「改行+インデント」を1単位にする独自事前分割（domain方式）を採用。
- **構造タグ**: 設計図の7種に加え、BP_RECIPE / BP_LOOT / RP_PARTICLE / SCRIPT_DTS / DOC / MOLANG_DOC 等を追加。
  開きタグ・閉じタグとも特殊トークン。
- **公式JSON Schema**: bedrock-samples の `metadata/json_schemas` に公式スキーマが含まれているため、
  評価指標「スキーマ準拠率」はこれを使える。
