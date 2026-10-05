"""学習済みモデルを「リリース」としてGitに記録し、タグを付ける。

    uv run python -m llm.release --run v1 --name mossanai-v1 [--note "変更点"] [--commit]

releases/<name>/ に以下をまとめる（safetensorsはGit LFS）:
  adapter_model.safetensors, adapter_config.json … LoRAの重み
  train_args.json, train_log.txt                 … 学習設定とログ
  eval/*.summary.json                            … 評価結果（out/eval/<name>_*.summary.json）
  release.json                                   … ベースモデルの版・学習データのハッシュ・作成日時
--commit で git commit と git tag <name> まで行う。
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from llm.fim import PROJECT_ROOT


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
                          encoding="utf-8").stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="out/lora/<run>")
    ap.add_argument("--name", required=True, help="リリース名（Ollamaのモデル名・Gitタグにも使う）")
    ap.add_argument("--checkpoint", default="best", help="best / last")
    ap.add_argument("--note", default="")
    ap.add_argument("--commit", action="store_true")
    a = ap.parse_args()

    # 学習データが未コミットだと、release.json の code_commit から学習データを再現できなくなる
    dirty = git("status", "--porcelain", "--", "data/sft", "llm", "bedrock_tools")
    if dirty:
        raise SystemExit(f"学習データ/コードに未コミットの変更があります。先にコミットしてください:\n{dirty}")

    run_dir = PROJECT_ROOT / "out" / "lora" / a.run
    ckpt = run_dir / a.checkpoint
    dest = PROJECT_ROOT / "releases" / a.name
    if dest.exists():
        raise SystemExit(f"{dest} は既に存在します。別の名前にしてください（リリースは上書きしない）")
    dest.mkdir(parents=True)
    for f in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy(ckpt / f, dest / f)
    shutil.copy(run_dir / "train_args.json", dest / "train_args.json")
    if (run_dir / "log.txt").exists():
        shutil.copy(run_dir / "log.txt", dest / "train_log.txt")

    evals = sorted((PROJECT_ROOT / "out" / "eval").glob(f"{a.name}_*.summary.json"))
    if evals:
        (dest / "eval").mkdir()
        for e in evals:
            shutil.copy(e, dest / "eval" / e.name.removeprefix(f"{a.name}_"))

    train_args = json.loads((run_dir / "train_args.json").read_text(encoding="utf-8"))
    data_files = [PROJECT_ROOT / p for p in train_args["data"] + train_args.get("val", [])]
    sources = json.loads((PROJECT_ROOT / "sources.json").read_text(encoding="utf-8"))
    info = {
        "name": a.name,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "note": a.note,
        "base_model": sources["base_model"],
        "bedrock_samples": sources["bedrock_samples"],
        "run": a.run,
        "checkpoint": a.checkpoint,
        "code_commit": git("rev-parse", "HEAD"),
        "data_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in data_files},
        "eval": {e.name.removeprefix(f"{a.name}_").removesuffix(".summary.json"):
                 json.loads(e.read_text(encoding="utf-8")) for e in evals},
    }
    (dest / "release.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"リリースを作成しました: {dest}")

    if a.commit:
        git("add", str(dest), *(str(e) for e in evals))
        git("commit", "-m", f"リリース {a.name}" + (f": {a.note}" if a.note else "")
            + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
        git("tag", "-a", a.name, "-m", f"{a.name} {a.note}".strip())
        print(f"コミットしてタグ {a.name} を付けました")


if __name__ == "__main__":
    main()
