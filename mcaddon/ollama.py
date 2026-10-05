"""ローカルの Ollama API クライアント（標準ライブラリのみ）。"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _post(path: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(f"{HOST.rstrip('/')}{path}", data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:  # サーバーは動いているがエラー（メモリ不足・モデル無しなど）
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama がエラーを返しました（HTTP {e.code}）: {detail}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Ollama に接続できません（{HOST}）。Ollama を起動してください: {e}") from e


def chat(model: str, user: str, temperature: float = 0.2, num_ctx: int = 8192, timeout: int = 600,
         max_tokens: int | None = None) -> str:
    options = {"temperature": temperature, "num_ctx": num_ctx}
    if max_tokens:
        options["num_predict"] = max_tokens
    res = _post("/api/chat", {"model": model, "stream": False, "options": options,
                              "messages": [{"role": "user", "content": user}]}, timeout)
    return res["message"]["content"]


def fim(model: str, prefix: str, suffix: str, max_tokens: int = 128, temperature: float = 0.0,
        timeout: int = 120) -> tuple[str, float]:
    """FIM補完（VSCode拡張と同じ /api/generate の suffix 指定）。(補完テキスト, 生成秒数)"""
    res = _post("/api/generate", {"model": model, "prompt": prefix, "suffix": suffix, "stream": False,
                                  "options": {"temperature": temperature, "num_predict": max_tokens}}, timeout)
    return res["response"], res.get("total_duration", 0) / 1e9
