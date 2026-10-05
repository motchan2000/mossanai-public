# Minecraft Bedrock Add-on特化LLM 設計図

## 0. プロジェクトの前提整理

### 0.1 ゴールの再定義
「軽量・高速・高精度」を実現する鍵は **スクラッチ学習そのもの** ではなく **スコープを絞ること** です。以下の2トラックを両方視野に入れて設計します。

| トラック | 内容 | 向いている場合 |
|---|---|---|
| **Aトラック（学習用フルスクラッチ）** | STEP1〜12を全部自分で実装 | Transformerの仕組みを理解したい／将来もっと自由にカスタムしたい |
| **Bトラック（実用最短ルート）** | 既存の小型OSSモデル（Qwen2.5-0.5B/1.5B、StableLM 2 1.6B、Gemma 2 2Bなど）をベースにFine-tuning | とにかく早く「使えるAdd-on特化LLM」が欲しい |

**推奨**: まずAトラックを小規模（数百万〜数千万パラメータ）で最後まで一通り動かして仕組みを体得し（STEP1〜8）、実用フェーズ（STEP9〜12）はBトラックに切り替える。ゼロから事前学習して実用レベルの言語理解を得るには数十億トークン級のデータと数百〜数千GPU時間が必要で、個人環境では現実的でないためです。

### 0.2 ドメインスコープの定義（これが精度と軽量化の要）
Bedrock Add-on開発の知識を以下に限定し、それ以外（一般雑談・他ゲーム知識・他言語プログラミング全般）は扱わないと割り切ります。

- **Behavior Pack**: entities, items, blocks, recipes, loot_tables, spawn_rules, trading, feature/feature_rules
- **Resource Pack**: models, textures, animations, animation_controllers, particles, sounds, render_controllers, attachables
- **Scripting API**: `@minecraft/server`, `@minecraft/server-ui`, `@minecraft/server-admin`, `@minecraft/server-net`（TypeScript/JavaScript）
- **Molang**: 式構文、query関数、variable設計
- **manifest.json / パッケージング**: UUID構造、dependencies、モジュールタイプ、pack互換性（min_engine_version）
- **JSON Schema整合性**: バージョンごとのformat_version差分

この境界をタスク定義・データ収集・評価指標すべてに一貫して適用します。

---

## 1. 全体アーキテクチャ

```
[Tokenizer] → [Embedding + Positional] → [Transformer Block × N] → [LM Head] → [次トークン予測]
```

- 言語: Python 3.11+
- フレームワーク: PyTorch 2.x + CUDA 12.x
- 参考実装: nanoGPT（Andrej Karpathy）をベースに、Bedrock特化のトークナイザ・データパイプラインを追加する構成が最も現実的です。

### モデルサイズの目安

| 用途 | パラメータ数 | 層数 | d_model | head数 | context長 | 想定VRAM(学習) |
|---|---|---|---|---|---|---|
| 学習用ミニマム(STEP1〜8) | 3〜10M | 4〜6 | 128〜256 | 4〜8 | 256〜512 | 4GB程度 |
| ドメイン特化・実用最小(STEP9〜12目標) | 60〜125M | 8〜12 | 512〜768 | 8〜12 | 1024〜2048 | 8〜16GB |
| Bトラック(既存モデルFine-tune) | 500M〜1.6B | - | - | - | 4096+ | 12〜24GB(QLoRA) |

Add-on開発は「JSON生成」「短いTypeScriptスニペット生成」「構文チェック」が中心タスクなので、汎用チャットLLMほどの規模は不要です。125M前後でも、スコープを絞ったデータで学習すれば実用的なJSON/コード補完精度は十分狙えます。

---

## STEP 1: Python + PyTorch理解

- **目的**: Tensor操作、autograd、`nn.Module`、DataLoader、学習ループの一連の流れを体に入れる。
- **やること**:
  - Tensorの次元操作（reshape, transpose, matmul, broadcasting）
  - `torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.LayerNorm` を素朴なMLPで組んでみる
  - `optimizer.zero_grad() → loss.backward() → optimizer.step()` の意味を説明できるようにする
  - GPU転送 (`.to("cuda")`) とAMP (`torch.cuda.amp.autocast`) を使った学習速度比較
- **完了基準**: MNIST程度のトイタスクを自力でスクラッチ実装できる。

---

## STEP 2: Tokenizerの自作（ドメイン特化の最重要ポイント）

汎用BPE（GPT-2 tokenizerなど）はJSON構造やMolang式のトークン効率が悪く、無駄にcontext長を消費します。専用トークナイザで語彙を最適化することが「軽量・高速」の核心です。

### 設計方針
- ベースアルゴリズム: **Byte-level BPE**（`tokenizers`ライブラリ / Hugging Face製、Rust実装で高速）
- 語彙サイズ: 8,000〜16,000（汎用LLMの50,000超に比べ大幅に小さくでき、Embedding層が軽くなる）
- **ドメイン特化の特殊トークンを事前定義**:
  - 構造トークン: `<BP_ENTITY>`, `<BP_ITEM>`, `<BP_BLOCK>`, `<RP_MODEL>`, `<RP_ANIM>`, `<MANIFEST>`, `<SCRIPT_TS>`
  - JSON構文トークン: `{`, `}`, `[`, `]`, `:`, `,`, `"` を高頻度トークンとして優先学習
  - Bedrock頻出キー: `"format_version"`, `"minecraft:entity"`, `"components"`, `"minecraft:behavior."`, `"minecraft:on_"` などをサブワードでなく1トークン化できるよう学習コーパスに大量投入
  - Molang頻出関数: `query.`, `variable.`, `math.`, `v.`, `q.` を短縮効率化
- **学習コーパス**: STEP9で集める生データ（後述）をそのままBPE学習にかける（BPEは教師なしなので生データで十分）

### 実装
```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["corpus/bedrock_all.txt"],
    vocab_size=12000,
    min_frequency=2,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>",
                     "<BP_ENTITY>", "<BP_ITEM>", "<BP_BLOCK>",
                     "<RP_MODEL>", "<RP_ANIM>", "<MANIFEST>", "<SCRIPT_TS>"]
)
tokenizer.save_model("tokenizer/")
```
- **完了基準**: 実際のentity JSONファイルをエンコード→デコードして完全復元できる。かつ汎用GPT-2 tokenizerよりトークン数が同一ファイルで20〜40%少ないことを確認する（軽量化の定量根拠）。

---

## STEP 3: Embeddingの実装

- **Token Embedding**: `nn.Embedding(vocab_size, d_model)`
- **Positional Embedding**: 学習可能な絶対位置埋め込み、または将来の拡張性を考え **RoPE (Rotary Positional Embedding)** を採用推奨（context長を後から伸ばしやすく、JSON構造のネスト位置関係の学習に有利）
- **完了基準**: `[batch, seq_len] → [batch, seq_len, d_model]` の変換をテストコードで確認。

---

## STEP 4: Self-Attentionの実装

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head)
        q, k, v = qkv.unbind(dim=2)
        # PyTorch 2.x: Flash Attention相当を自動選択させる
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True
        )
        out = out.transpose(1,2).reshape(B, T, C)
        return self.proj(out)
```
- `scaled_dot_product_attention` を使うことでFlash Attention相当の高速化を自動的に得られる（推論速度＝「軽量・高速」に直結）。
- **完了基準**: causal maskが正しく機能し、未来トークンを参照していないことをテストで検証。

---

## STEP 5: Transformer Blockの実装

```
x = x + SelfAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```
- Pre-LN構成（学習安定性が高く小規模モデルでも収束しやすい）
- MLP: `Linear(d_model, 4*d_model) → GELU → Linear(4*d_model, d_model)`
- **完了基準**: Blockを10層重ねても勾配爆発/消失なく forward/backward が通る。

---

## STEP 6: GPT型LLMの完成

```
Input IDs → Token Emb + Pos Emb → [Transformer Block] × N → LayerNorm → LM Head(Linear) → Logits
```
- 重み共有（Token EmbeddingとLM Headの重みタイイング）でパラメータ数を削減 → 軽量化に直結
- **完了基準**: ランダム初期化状態でforward passが通り、`loss = F.cross_entropy(logits, targets)` が計算できる。

---

## STEP 7: 小規模データで学習

- まず**汎用の小さいテキスト**（例: シェイクスピアデータセットなど定番のトイデータ）で学習ループ自体が正しく機能するか検証。
- ここでBedrockデータを使わないのは、パイプラインのバグとデータ品質の問題を切り分けるためです。
- **完了基準**: lossがちゃんと下がり、簡単な文章の続きがそれらしく生成される。

---

## STEP 8: 文章生成（推論ロジック）

- Greedy / Top-k / Top-p (nucleus) samplingを実装
- Bedrock向けには **JSON構文を壊さない制約付き生成** が重要になるため、この段階で以下を仕込んでおくと後が楽:
  - 生成中に閉じ括弧の対応をチェックする簡易バリデータ
  - 将来的にはGrammar-constrained decoding（例: `outlines`, `llguidance` ライブラリでJSON Schema制約付き生成）を組み込む前提で設計しておく
- **完了基準**: プロンプトから一貫した文章が生成できる。

---

## STEP 9: 大規模データセットでPretraining

### データ収集ソース（Bedrock特化）
| ソース | 内容 | 取得方法 |
|---|---|---|
| Microsoft公式 `bedrock-samples` (GitHub) | Vanillaのbehavior/resource pack全JSON | git clone |
| Bedrock Scripting API公式ドキュメント | `@minecraft/server` 全クラス・メソッド定義とサンプル | 公式ドキュメントサイトのクロール(利用規約確認の上) |
| Molang公式ドキュメント | 構文・関数リファレンス | 同上 |
| ゲーム本体のvanillaデータ | 実際のentity/item/block定義(最新format_version) | ゲームインストールディレクトリから抽出 |
| コミュニティAdd-on(GitHub, OSSライセンス限定) | 実践的なentity構成・スクリプト例 | GitHub検索APIで `manifest.json` を含むリポジトリを収集、ライセンス確認必須 |
| 自作の合成データ | 「〜な敵Mobを作って」→対応JSON、のペアをテンプレートから自動生成 | スクリプトで大量生成 |

**注意**: ライセンス不明なAdd-onコードは学習データに含めない。公式ドキュメント・公式サンプル・自作合成データを主軸にし、コミュニティデータはMIT/Apache等の明示的にOSSなものだけに限定する。

### 学習フォーマット設計
生データをそのまま流すのではなく、**構造タグ付きテキスト**として整形:
```
<MANIFEST>{"format_version": 2, "header": {...}}</MANIFEST>
<BP_ENTITY>{"format_version": "1.21.0", "minecraft:entity": {...}}</BP_ENTITY>
<SCRIPT_TS>import { world, system } from "@minecraft/server";
world.afterEvents.entitySpawn.subscribe((event) => { ... });</SCRIPT_TS>
```
これにより、モデルが「今どの種類のファイルを書いているか」を明示的に学習でき、少ないパラメータ数でも構造理解の精度が上がります。

### 学習設定の目安
- Optimizer: AdamW (lr=3e-4, weight_decay=0.1, cosine decay + warmup)
- Batch: gradient accumulationで実効batch 128〜256相当
- Mixed precision (`bf16`) 必須（速度・メモリ両面で有利）
- **完了基準**: validation lossが継続的に下がり、held-outのJSONファイルの補完で構文エラー率が計測可能なレベルまで下がる。

---

## STEP 10: 専門データでFine-tuning

Pretrainingで「Bedrockの語彙・構文」を学ばせたら、Fine-tuningで「指示→成果物」の対応を学ばせます。

### Instruction形式の例
```
### Instruction:
夜だけ湧く、プレイヤーを追いかけて爆発する敵Mobのbehavior定義をJSONで書いて

### Response:
<BP_ENTITY>{ ... "minecraft:behavior.melee_attack": {...}, "minecraft:explode": {...} ... }</BP_ENTITY>
```
- タスク種類を分散させる: JSON生成／TypeScript生成／既存JSONのバグ修正／manifest.jsonのdependencies修正／Molang式の説明 など
- Bトラック(既存OSSモデル利用時)は **LoRA/QLoRA** で低コストFine-tuningするのが効率的（フルパラメータ更新よりVRAM消費が桁違いに少ない）
- **完了基準**: 未見の指示に対し、構文的に妥当なJSON/コードを一定割合以上生成できる。

---

## STEP 11: 量子化

| 手法 | 用途 | ツール |
|---|---|---|
| Post-training量子化 (int8/int4) | 学習後モデルをそのまま軽量化 | `bitsandbytes` |
| GPTQ / AWQ | 精度劣化を抑えた4bit量子化 | `auto-gptq`, `autoawq` |
| GGUF変換 | ローカル推論エンジン向け配布形式 | `llama.cpp` の `convert.py` |

- 目標: fp16比でモデルサイズを1/4程度に圧縮しつつ、JSON構文正当性などのタスク指標の劣化を数%以内に抑える。
- **完了基準**: 量子化前後でベンチマーク（後述の評価指標）を比較し、許容範囲の精度低下であることを確認。

---

## STEP 12: ローカルAIアプリ化

- 推論エンジン: `llama.cpp`（GGUF化済みモデル）または `Ollama`（Modelfileでカスタムモデル登録）
- UI選択肢:
  - CLIツール（`typer`等でコマンド化、`mcaddon-gen "夜行性の敵Mob"` のような使い方）
  - ローカルWeb UI（FastAPI + 簡易フロント、または既存のOpen WebUI連携）
  - VSCode拡張（Add-on開発中にインラインでJSON/TS補完を提示）
- 出力後処理: 生成JSONを実際に `jsonschema` でBedrockのスキーマ検証にかけ、壊れていれば警告表示する後処理レイヤーを必ず挟む（LLM単体の出力を無条件に信頼しない設計）

---

## 2. 評価指標の設計（ドメイン特化ならではの重要ポイント）

汎用LLMのPerplexityだけでなく、以下のBedrock固有指標を用意します。

1. **JSON構文妥当性率**: 生成物が正しいJSONとしてパースできる割合
2. **スキーマ準拠率**: `jsonschema` でBedrock公式スキーマに対しvalidateが通る割合
3. **format_version整合性**: 使用しているコンポーネントとformat_versionの組み合わせが実在するかのチェック
4. **Molang構文正当性**: 簡易パーサでMolang式が文法的に正しいか
5. **UUID重複チェック**: manifest.json内のUUIDが正しいUUID形式かつheader/modules間で整合しているか

これらを自動テストスイート化し、STEP9〜11の各段階で回帰テストとして走らせます。

---

## 3. 全体ロードマップ（時系列まとめ）

```
Phase 1（学習）   STEP1-8   … Transformerの仕組みを小規模データで体得
Phase 2（構築）   STEP9     … Bedrock特化データ収集 + Tokenizer最終化 + Pretraining
Phase 3（特化）   STEP10    … Instruction形式でFine-tuning
Phase 4（配布）   STEP11-12 … 量子化 + ローカルアプリ化 + 自動検証パイプライン
```

---

## 4. 想定される詰まりどころと対策

| 詰まりどころ | 対策 |
|---|---|
| Bedrockの公式データだけでは学習データ量が不足 | テンプレートベースの合成データ生成を積極活用（パラメータをランダムに変えたentity定義を大量生成） |
| format_versionの違いでデータが矛盾する | 学習データにformat_versionを明示タグ化し、バージョンごとに正しい構文を紐付けて学習 |
| ゼロから事前学習すると言語理解が弱くまとまりのない文章になる | Bトラック（既存小型OSSモデルのFine-tune）に切り替える、またはPretrainingデータに一般的な英語/日本語コーパスを一定割合(10〜20%)混ぜて言語能力の土台を維持する |
| 量子化で精度が大きく落ちる | AWQなど重要層を保護する量子化手法を優先し、加えてSTEP2の語彙圧縮で元々のモデルを小さく保つ |

---

以上が全体設計図です。次にすぐ着手できる粒度まで分解したい場合（例: STEP2のトークナイザ学習スクリプト一式や、STEP9のデータ収集スクリプト）は、対象のSTEPを指定してください。
