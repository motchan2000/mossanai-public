"""STEP9: Mojang公式 bedrock-samples を、バージョン固定でテキストファイルだけ取得する。

画像・音声などのバイナリは学習に使わないので sparse-checkout で除外する。
"""
import argparse
import subprocess
from pathlib import Path

REPO = "https://github.com/Mojang/bedrock-samples.git"
DEFAULT_TAG = "v1.26.50.4"
PATTERNS = ["*.json", "*.js", "*.ts", "*.md", "*.lang", "*.material"]

here = Path(__file__).parent


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default=DEFAULT_TAG)
    a = p.parse_args()

    dest = here / "raw" / f"bedrock-samples-{a.tag}"
    if dest.exists():
        print(f"既に取得済み: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", "--depth", "1", "--branch", a.tag, "--filter=blob:none", "--no-checkout", REPO, str(dest))
    run("git", "sparse-checkout", "set", "--no-cone", *PATTERNS, cwd=dest)
    run("git", "checkout", a.tag, cwd=dest)
    print(f"取得完了: {dest}")


if __name__ == "__main__":
    main()
