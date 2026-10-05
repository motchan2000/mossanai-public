"""ユーザーのコード単位から「それを頼むときの日本語の依頼文」を逆算する（教師: ローカルの qwen3-coder）。

    uv run python -m llm.backtranslate [--model qwen3-coder:30b] [--limit N]

data/user/units.jsonl の型チェックOKの単位について、data/user/backtrans.jsonl に追記していく
（中断しても、済んだ単位は飛ばして再開できる）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

from llm.fim import PROJECT_ROOT
from mcaddon.ollama import chat

UNITS = PROJECT_ROOT / "data" / "user" / "units.jsonl"
OUT = PROJECT_ROOT / "data" / "user" / "backtrans.jsonl"

PROMPT = """以下は Minecraft 統合版アドオンの Script API（JavaScript）のコードの一部です（ファイル: {file}）。

```js
{imports}

{code}
```

このコードを AI に書いてもらうとしたら、アドオン開発者はどんな依頼をするでしょうか。
開発者がチャットで AI に頼むときの、自然な日本語の依頼文を1つ書いてください。

条件:
- 1〜3文。口調はくだけていてよい（例: 「〜して」「〜を作りたい」）
- コードの目的と具体的な振る舞い（どのイベントで何をするか、扱う値、表示する文言など）が伝わるように
- 関数名やエクスポート名を指定したほうが自然なら含める
- コードそのもの、説明、前置きは書かない。依頼文だけを出力する"""


def unit_id(u: dict) -> str:
    return f"{u['pack']}:{u['file']}:{u['start']}"


def clean(text: str) -> str:
    text = text.strip().strip("「」\"'")
    text = re.sub(r"^(依頼文|依頼)[:：]\s*", "", text)
    text = re.sub(r"[」』]\s*[「『]", "。", text)  # 複数の発言に分けて書く癖を1つの依頼文にまとめる
    return text.split("\n\n")[0].strip().rstrip("」』")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-coder:30b")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    units = [json.loads(l) for l in UNITS.read_text(encoding="utf-8").splitlines() if l]
    units = [u for u in units if u["tsc_ok"]]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["id"] for l in OUT.read_text(encoding="utf-8").splitlines() if l}
    todo = [u for u in units if unit_id(u) not in done][: a.limit]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)

    t0 = time.time()
    with open(OUT, "a", encoding="utf-8", newline="\n") as f:
        for n, u in enumerate(todo, 1):
            prompt = PROMPT.format(file=u["file"], imports=u["imports"], code=u["code"])
            try:
                req = clean(chat(a.model, prompt, temperature=0.3, num_ctx=8192, max_tokens=250, timeout=600))
            except Exception as e:  # 1件の失敗で全体を止めない
                print(f"  失敗 {unit_id(u)}: {e}", flush=True)
                continue
            if len(req) < 8:
                continue
            f.write(json.dumps({"id": unit_id(u), "request": req}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n <= 3:
                rate = (time.time() - t0) / n
                print(f"[{n}/{len(todo)}] {rate:.1f}s/件 残り約{rate * (len(todo) - n) / 60:.0f}分 | {req[:70]}",
                      flush=True)


if __name__ == "__main__":
    main()
