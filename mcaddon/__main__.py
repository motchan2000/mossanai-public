"""日本語の指示からアドオンのファイルを生成し、検証→自己修正してパックに書き込む。

    uv run python -m mcaddon gen "夜だけ湧く、プレイヤーを追いかけて爆発する敵Mob。IDはmy:bomber" --pack ./packs/MyAddon
    uv run python -m mcaddon gen "..." --pack ./packs/MyAddon --dry-run   # 書き込まずに表示だけ
    uv run python -m mcaddon gen "/menu でメニューを開くコマンド" --model mossanai-v4 --runtime   # Script API

--pack のフォルダに *_BP / *_RP が無ければ雛形（manifest込み）を自動で作る。

流れは2種類（--flow。既定の auto は mossanai-v1〜v3 なら json、それ以外は script）:
  json   … v1/v2 用。指示をそのまま渡し、JSON を検証→修正
  script … v4 以降・汎用モデル用。型定義の RAG + お手本を添えて生成し、スクリプトは型チェック（--runtime なら
           BDS の実行テストも）で自己修正、JSON は検証→修正。学習時と同じプロンプト（mcaddon.script_gen）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from bedrock_tools.check import check_file
from bedrock_tools.jsonc import dump, parse_lenient
from bedrock_tools.manifest import scaffold
from bedrock_tools.schema_validator import SchemaValidator
from llm.build_gen_data import parse_answer
from llm.build_repair_data import USER_TEMPLATE
from llm.chat import code_block, extract_code_block
from mcaddon.ollama import chat

DEFAULT_MODEL = os.environ.get("MCADDON_MODEL", "mossanai-v1")
JSON_ONLY_MODELS = ("mossanai-v1", "mossanai-v2", "mossanai-v3")

Files = list[tuple[str, str, list[str]]]  # (パス, 内容, 残った問題)


def validate(path: str, text: str, validator: SchemaValidator, tmp_dir: Path) -> tuple[list[str], list[str]]:
    tmp = tmp_dir / ("manifest.json" if path.endswith("manifest.json") else Path(path).name)
    tmp.write_text(text, encoding="utf-8")
    rep = check_file(tmp, validator)
    # 「何も起きない」系の警告は実害があるので修正対象に含める
    actionable = [w for w in rep.warnings if "何も起こしません" in w or "使われていないフィールド" in w]
    return rep.errors, actionable


def repair(model: str, path: str, text: str, problems: list[str]) -> str:
    user = USER_TEMPLATE.format(path=path, code=code_block(text), issues="\n".join(f"- {p}" for p in problems[:8]))
    return extract_code_block(chat(model, user))


def resolve_pack(pack: Path, name: str, with_script: bool = False) -> tuple[Path, Path]:
    bps = sorted(pack.glob("*_BP")) if pack.exists() else []
    rps = sorted(pack.glob("*_RP")) if pack.exists() else []
    if bps and rps:
        return bps[0], rps[0]
    pack.mkdir(parents=True, exist_ok=True)
    bp, rp = scaffold(pack, name, with_script=with_script)
    if with_script:
        (bp / "scripts" / "main.js").unlink()  # 雛形の main.js は生成したものに置き換える
    return bp, rp


def fix_json_files(a, files: list[tuple[str, str]], validator: SchemaValidator, tmp_dir: Path) -> Files:
    final: Files = []
    for path, text in files:
        errors, warns = validate(path, text, validator, tmp_dir)
        for round_ in range(a.max_repair):
            if not errors and not warns:
                break
            print(f"  {path}: 問題 {len(errors) + len(warns)} 件 → 修正 {round_ + 1} 回目", flush=True)
            text = repair(a.model, path, text, errors + warns)
            errors, warns = validate(path, text, validator, tmp_dir)
        doc, err = parse_lenient(text)
        if doc is not None:
            text = dump(doc)  # 書式をプロジェクト標準に揃える
        final.append((path, text, errors + warns))
    return final


def script_files(bp: Path, scripts: dict[str, str]) -> Files:
    """スクリプトの書き込み内容。manifest の entry（scripts/main.js）から読まれるようにし、依存モジュールを揃える。"""
    from bedrock_tools.runtime_check import _entry_script, _module_versions

    scripts, entry = _entry_script(scripts)
    if entry != "scripts/main.js":
        scripts = {("scripts/main.js" if p == entry else p): c for p, c in scripts.items()}
    out: Files = [(f"BP/{p}", c, []) for p, c in scripts.items()]
    man = json.loads((bp / "manifest.json").read_text(encoding="utf-8"))
    if not any(m.get("type") == "script" for m in man.get("modules", [])):
        print("  注意: manifest に script モジュールがありません（entry に scripts/main.js を指定してください）")
    have = {d.get("module_name") for d in man.get("dependencies", [])}
    missing = {m: v for m, v in _module_versions(scripts).items() if m not in have}
    if missing:
        man.setdefault("dependencies", []).extend({"module_name": m, "version": v} for m, v in missing.items())
        out.append(("BP/manifest.json", dump(man), []))
        print(f"  manifest の依存に追加: {', '.join(missing)}")
    return out


def write_files(a, final: Files, scripts: dict[str, str] | None = None) -> int:
    ok = all(not p for _, _, p in final)
    for path, text, problems in final:
        mark = "✓" if not problems else "✗"
        print(f"{mark} {path}" + ("" if not problems else "\n    " + "\n    ".join(problems[:5])))
    if a.dry_run:
        for path, text, _ in final:
            print(f"\n### {path}\n{text}")
        return 0 if ok else 1

    bp, rp = resolve_pack(a.pack, a.pack.name, with_script=bool(scripts))
    if scripts:
        final = final + script_files(bp, scripts)
    for path, text, problems in final:
        root, _, rel = path.partition("/")
        dest = (bp if root.upper() == "BP" else rp if root.upper() == "RP" else bp) / rel
        if problems and not a.force:
            print(f"  問題が残っているため書き込みません: {path}（--force で書き込み、--dry-run で内容確認）")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not a.overwrite and dest.name != "manifest.json":
            print(f"  既に存在するためスキップ: {dest}（--overwrite で上書き）")
            continue
        dest.write_text(text.rstrip("\n") + "\n", encoding="utf-8")
        print(f"  書き込み: {dest}")
    return 0 if ok else 1


def cmd_gen_json(a) -> int:
    validator = SchemaValidator()
    tmp_dir = Path(tempfile.mkdtemp(prefix="mcaddon_"))
    print(f"モデル {a.model} で生成中…", flush=True)
    answer = chat(a.model, a.instruction, temperature=a.temperature)
    files = parse_answer(answer)
    if not files:
        print("ファイルを読み取れませんでした。モデルの出力:\n" + answer)
        return 1
    return write_files(a, fix_json_files(a, files, validator, tmp_dir))


def cmd_gen_script(a) -> int:
    from llm.eval_user import extract
    from mcaddon.script_gen import generate

    validator = SchemaValidator()
    tmp_dir = Path(tempfile.mkdtemp(prefix="mcaddon_"))
    print(f"モデル {a.model} で生成中…（型定義の参照・型チェック{'・実行テスト' if a.runtime else ''}で最大 "
          f"{a.max_repair} 回の自己修正）", flush=True)
    res = generate(a.model, a.instruction, extract, max_repair=a.max_repair, compact=True, runtime=a.runtime)
    from mcaddon.script_gen import json_path

    # JSON は自己修正後の最終版（検証は生成ループで済んでいるが、残った問題の表示と書式の統一のため再検証する）
    files = [(json_path(b, i), b) for i, b in enumerate(res.jsons) if json_path(b, i).count("/") > 1]
    if not files and not res.scripts:
        print("ファイルを読み取れませんでした。モデルの出力:\n" + res.output)
        return 1
    final = fix_json_files(a, files, validator, tmp_dir)
    if res.scripts:
        mark = "✓" if not res.errors else "✗"
        print(f"{mark} スクリプト {len(res.scripts)} ファイル（自己修正 {res.rounds} 回）"
              + ("" if not res.errors else "\n    " + "\n    ".join(res.errors[:5])))
        if a.dry_run:
            print("\n" + "\n\n".join(f"### {p}\n{c}" for p, c in res.scripts.items()))
        if res.errors and not a.force:
            print("  スクリプトにエラーが残っているため書き込みません（--force で書き込み）")
            write_files(a, final)
            return 1
    return write_files(a, final, res.scripts or None)


def cmd_gen(a) -> int:
    flow = a.flow if a.flow != "auto" else ("json" if a.model.split(":")[0] in JSON_ONLY_MODELS else "script")
    return cmd_gen_script(a) if flow == "script" else cmd_gen_json(a)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="mcaddon")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="指示からファイルを生成")
    g.add_argument("instruction")
    g.add_argument("--pack", type=Path, default=Path("packs/MyAddon"))
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--max-repair", dest="max_repair", type=int, default=2)
    g.add_argument("--temperature", type=float, default=0.2)
    g.add_argument("--dry-run", dest="dry_run", action="store_true")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--force", action="store_true", help="検証で問題が残ったファイルも書き込む")
    g.add_argument("--flow", choices=["auto", "json", "script"], default="auto")
    g.add_argument("--runtime", action="store_true", help="スクリプトを BDS で読み込ませて実行時エラーも確認（数十秒かかる）")
    a = ap.parse_args(argv)
    return cmd_gen(a)


if __name__ == "__main__":
    sys.exit(main())
