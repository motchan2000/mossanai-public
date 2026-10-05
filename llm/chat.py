"""指示データ用のChatML形式（Qwen系の標準）。"""
from __future__ import annotations

SYSTEM = ("あなたはMinecraft統合版（Bedrock Edition）のアドオン開発アシスタントです。"
          "公式仕様に沿った正しいJSONとスクリプトを書きます。")
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def chat_prompt(user: str, system: str = SYSTEM) -> str:
    return f"{IM_START}system\n{system}{IM_END}\n{IM_START}user\n{user}{IM_END}\n{IM_START}assistant\n"


def chat_completion(answer: str) -> str:
    return f"{answer}{IM_END}"


def user_text(prompt: str) -> str:
    """chat_prompt() で作ったプロンプトから user 部分を取り出す（Ollama API に渡す用）。"""
    return prompt.split(f"{IM_START}user\n", 1)[1].split(IM_END, 1)[0]


def code_block(text: str, lang: str = "json") -> str:
    return f"```{lang}\n{text.rstrip()}\n```"


def extract_code_block(text: str) -> str:
    """モデル出力から最初のコードブロックの中身を取り出す（無ければ全体）。"""
    if "```" not in text:
        return text.strip()
    body = text.split("```", 1)[1]
    body = body.split("\n", 1)[1] if "\n" in body else body
    return body.split("```", 1)[0].rstrip()
