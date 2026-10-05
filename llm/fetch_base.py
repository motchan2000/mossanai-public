"""sources.json に記録した版のベースモデルを取得する（Git管理外のファイルの復元用）。

    uv run python -m llm.fetch_base                      # 1.5B（補完用）
    uv run python -m llm.fetch_base --key base_model_7b  # 7B Instruct（CLI 生成用、約15GB）
"""
import argparse
import json

from huggingface_hub import snapshot_download

from llm.fim import PROJECT_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="base_model", help="sources.json のキー")
    src = json.loads((PROJECT_ROOT / "sources.json").read_text(encoding="utf-8"))[ap.parse_args().key]
    path = snapshot_download(src["repo"], revision=src["revision"], local_dir=PROJECT_ROOT / src["local_dir"],
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE*", "README.md"])
    print(path)


if __name__ == "__main__":
    main()
