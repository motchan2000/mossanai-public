"""Script API についての現状把握: 依頼をモデルに投げ、出力スクリプトを型チェックする。

    uv run python -m llm.probe_script --model mossanai-v2 qwen3-coder:30b
"""
import argparse
import re
import sys
import time

from bedrock_tools.script_check import check_scripts
from mcaddon.ollama import chat

PROMPTS = [
    "プレイヤーがワールドに参加したらチャットで「ようこそ」と表示するスクリプトを @minecraft/server で書いて。",
    "ダイヤの剣でエンティティを攻撃したら、そのエンティティに雷を落とすスクリプトを書いて。",
    "ワールド読み込み時にカスタムコマンド /heal を登録して、実行したプレイヤーの体力を全回復させて。",
]
CODE_RE = re.compile(r"```(?:javascript|js|typescript|ts)?\n(.*?)```", re.S)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["mossanai-v2"])
    ap.add_argument("--show", action="store_true", help="コード全文を表示")
    a = ap.parse_args()
    for model in a.model:
        print(f"######## {model}")
        for p in PROMPTS:
            t = time.time()
            out = chat(model, p + "（@minecraft/server 2.x の安定版APIで、JavaScriptで書いてください）",
                       temperature=0.0, max_tokens=1200)
            blocks = CODE_RE.findall(out)
            code = max(blocks, key=len) if blocks else ""
            issues = check_scripts({"scripts/main.js": code}) if code else None
            status = "コードなし" if issues is None else ("型チェックOK" if not issues else f"エラー{len(issues)}件")
            print(f"== {p[:30]}… [{status}] {time.time() - t:.0f}s")
            for i in (issues or [])[:4]:
                print(f"   {i}")
            if a.show and code:
                print(code)


if __name__ == "__main__":
    main()
