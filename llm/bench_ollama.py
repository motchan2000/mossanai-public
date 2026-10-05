"""Ollama 上のモデルの速度・メモリを測る（読み込み時間、プロンプト処理速度、生成速度、FIM 応答時間、VRAM）。

    uv run python -m llm.bench_ollama --model mossanai-v2
"""
import argparse
import json
import statistics
import subprocess
import sys
import time

from mcaddon.ollama import _post

PROMPT = "ルビーの剣を作って。IDはmy:ruby_sword。攻撃力は8、耐久値は500。"
MOB = "夜だけ湧く、プレイヤーを追いかけて攻撃する敵Mobを作って。IDはmy:ghoul。体力30。"
PREFIX = ('{\n  "format_version": "1.26.0",\n  "minecraft:entity": {\n    "components": {\n'
          '      "minecraft:health": {\n        "value": 20,\n        ')
SUFFIX = '\n      },\n      "minecraft:movement": {\n        "value": 0.25\n      }\n    }\n  }\n}'


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mossanai-v2")
    a = ap.parse_args()
    subprocess.run(["ollama", "stop", a.model], capture_output=True)

    t = time.time()
    r = _post("/api/generate", {"model": a.model, "prompt": "", "keep_alive": "10m"}, 300)  # 読み込みのみ
    print(f"読み込み時間: {time.time() - t:.1f} 秒（load_duration {r.get('load_duration', 0) / 1e9:.1f} 秒）")
    ps = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout
    print(ps.strip())

    for name, prompt in (("アイテム生成", PROMPT), ("Mob生成", MOB)):
        r = _post("/api/chat", {"model": a.model, "stream": False, "options": {"temperature": 0},
                                "messages": [{"role": "user", "content": prompt}]}, 600)
        pe = r["prompt_eval_count"] / max(r["prompt_eval_duration"] / 1e9, 1e-9)
        ev = r["eval_count"] / (r["eval_duration"] / 1e9)
        print(f"{name}: 出力 {r['eval_count']} トークン / 合計 {r['total_duration'] / 1e9:.1f} 秒 "
              f"（生成 {ev:.0f} トークン/秒、プロンプト処理 {pe:.0f} トークン/秒）")

    lat = []
    for _ in range(10):
        r = _post("/api/generate", {"model": a.model, "prompt": PREFIX, "suffix": SUFFIX, "stream": False,
                                    "options": {"temperature": 0, "num_predict": 32}}, 120)
        lat.append(r["total_duration"] / 1e6)
    print(f"FIM補完（1行）: 中央値 {statistics.median(lat):.0f} ms / 最大 {max(lat):.0f} ms （10回）")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print(f"GPUメモリ（全体）: {smi}")
    subprocess.run(["ollama", "stop", a.model], capture_output=True)


if __name__ == "__main__":
    main()
