"""Ollamaに登録したモデルの簡易動作確認（終了トークンで止まるか・既定値・FIM・速度）。

    uv run python -m llm.smoke_ollama --models mossanai-v1 mossanai-v2
"""
import argparse
import re
import sys

from mcaddon.ollama import _post, fim

PROMPTS = [
    "ルビーの剣を作って。IDはmy:ruby_sword。攻撃力は8、耐久値は500。",
    "夜だけ湧く、プレイヤーを追いかけて攻撃する敵Mobを作って。IDはmy:ghoul。体力30。",  # 速度・攻撃力は未指定
]
PREFIX = ('{\n  "format_version": "1.26.0",\n  "minecraft:entity": {\n    "components": {\n'
          '      "minecraft:health": {\n        "value": 20,\n        ')
SUFFIX = ('\n      },\n      "minecraft:movement": {\n        "value": 0.25\n      }\n    }\n  }\n}')


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mossanai-v1"])
    a = ap.parse_args()

    for model in a.models:
        print(f"===== {model}")
        for prompt in PROMPTS:
            r = _post("/api/chat", {"model": model, "stream": False,
                                    "options": {"temperature": 0, "num_predict": 3000},
                                    "messages": [{"role": "user", "content": prompt}]}, 900)
            c = r["message"]["content"]
            rate = r["eval_count"] / (r["eval_duration"] / 1e9)
            files = re.findall(r"### (\S+)", c)
            move = re.search(r'"minecraft:movement": \{\s*"value": ([\d.]+)', c)
            dmg = re.search(r'"minecraft:attack": \{\s*"damage": (\d+)', c)
            print(f"  {prompt[:20]}… tokens={r['eval_count']} ({rate:.0f} tok/s) stop={r.get('done_reason')}"
                  f" 末尾={c[-25:]!r}")
            print(f"    files={files} movement={move and move.group(1)} attack={dmg and dmg.group(1)}")
        fim(model, PREFIX, SUFFIX, max_tokens=32)  # 読み込み分を除くため2回目を測る
        text, sec = fim(model, PREFIX, SUFFIX, max_tokens=32)
        print(f"  FIM: {text!r} ({sec * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
