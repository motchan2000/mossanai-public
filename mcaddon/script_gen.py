"""Script API の生成ループ: RAG で参照付き生成 → tsc 型チェック（+ JSON の検証・BDS 実行テスト）→ エラーとヒントを返して修正。"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from bedrock_tools.script_check import check_scripts
from mcaddon.ollama import chat
from mcaddon.rag import build_context, repair_hints

INSTRUCTIONS = ("上の型定義と要点に従って、Minecraft 統合版の Script API（@minecraft/server 2.x 安定版）で実装してください。"
                "答えはそれだけで動くように完結させ、存在しない自作モジュールを import しないでください。"
                "スクリプトは ```js のコードブロックで、複数ファイルに分ける場合は各ブロックの直前に「### scripts/ファイル名.js」と書いてください。"
                "JSON は依頼がアイテム・ブロック・エンティティを新しく作ることを求めている場合だけ書き"
                "（スクリプトだけの依頼では書かない）、```json で、直前に「### BP/…」のようにパスを書いてください。")
JSON_FOLDERS = (("minecraft:block", "blocks"), ("minecraft:item", "items"), ("minecraft:entity", "entities"))


@dataclass
class GenResult:
    output: str  # 最初の出力（説明を含む）。最終版は scripts / jsons を使う

    scripts: dict[str, str]
    errors: list[str]
    rounds: int
    history: list[str] = field(default_factory=list)
    # 各ラウンドで検査したコードとそのエラー（修正の教材づくり用）。attempts[i] = (scripts, jsons, errors)
    attempts: list[tuple[dict[str, str], list[str], list[str]]] = field(default_factory=list)
    jsons: list[str] = field(default_factory=list)  # 最終版の JSON 本文（ブロック・アイテム・エンティティ等）


def json_path(body: str, i: int = 0) -> str:
    """JSON 本文から BP 内のパスを推定する（例: BP/blocks/lamp.json）。"""
    from bedrock_tools.jsonc import parse_lenient

    doc, _ = parse_lenient(body)
    for key, folder in JSON_FOLDERS:
        if isinstance(doc, dict) and key in doc:
            ident = str(doc[key].get("description", {}).get("identifier", f"gen_{i}"))
            return f"BP/{folder}/{ident.split(':')[-1]}.json"
    return f"BP/gen_{i}.json"


def check_jsons(jsons: list[str]) -> list[str]:
    """JSON の構文・スキーマ・意味の検証。旧方式（minecraft:custom_components）はゲームで読めないのでエラー扱い。"""
    from bedrock_tools.check import check_file
    from bedrock_tools.schema_validator import SchemaValidator

    if not jsons:
        return []
    tmp = Path(tempfile.mkdtemp(prefix="gen_json_"))
    validator = SchemaValidator()
    errors = []
    for i, body in enumerate(jsons):
        path = json_path(body, i)
        f = tmp / Path(path).name
        f.write_text(body, encoding="utf-8")
        rep = check_file(f, validator)
        # 「旧形式の format_version」の警告は使わない（ブロックのスキーマが 1.26.20 版しかなく、ゲームで動く 1.21.90 も旧形式になる）
        problems = rep.errors + [w for w in rep.warnings if "旧方式" in w]
        errors += [f"{path}: {p}" for p in problems]
    return errors


def check(scripts: dict[str, str], runtime: bool, jsons: list[str] | None = None) -> list[str]:
    """JSON の検証 + 型チェック → （通れば）BDS で実行して実行時エラー。エラー文字列のリストを返す。
    jsons（出力の JSON 本文）のブロック・アイテム・エンティティも一緒に読み込ませる（カスタムコンポーネントの紐付け確認）。"""
    errors = check_jsons(jsons or []) + ([str(i) for i in check_scripts(scripts)] if scripts else [])
    if not errors and runtime and scripts:
        from bedrock_tools.runtime_check import pack_json_files, run_pack

        rt = run_pack(scripts, json_files=pack_json_files(jsons or []))
        errors = [f"実行時エラー: {e}" for e in rt.errors] if rt.started else ["実行時エラー: サーバーが起動しませんでした"]
    return errors


def gen_context(request: str, compact: bool = False, examples: bool = True,
                exclude_packs: set[str] | None = None) -> str:
    """compact=True は mossanai 用（学習時と同じ小さい文脈: 早見表 + 関係する型定義を少し）。
    examples=True は例示RAG（似た依頼の自己完結したお手本を2つ添える）。exclude_packs のパックは使わない。"""
    if compact:
        from llm.build_v3_data import compact_context

        context = compact_context(request)
    else:
        context = build_context(request)
    if examples:
        from mcaddon.examples import examples_block

        context += "\n" + examples_block(request, k=2, exclude_packs=exclude_packs)
    return context


def format_scripts(scripts: dict[str, str], jsons: list[str] | None = None) -> str:
    parts = [f"### {json_path(b, i)}\n```json\n{b.rstrip()}\n```" for i, b in enumerate(jsons or [])]
    parts += [f"### {p}\n```{'ts' if p.endswith('.ts') else 'js'}\n{c.rstrip()}\n```" for p, c in scripts.items()]
    return "\n\n".join(parts)


def gen_prompt(context: str, request: str) -> str:
    return f"{context}\n# 依頼\n{request}\n\n{INSTRUCTIONS}"


def fix_prompt(context: str, request: str, scripts: dict[str, str], errors: list[str],
               jsons: list[str] | None = None) -> str:
    return (f"{context}\n# 依頼\n{request}\n\n# あなたが書いたコード\n{format_scripts(scripts, jsons)}\n\n"
            f"# 検証・型チェック・実行テスト（@minecraft/server 2.10.0）のエラー\n" + "\n".join(f"- {e}" for e in errors[:12]) +
            f"\n\n# ヒント\n{repair_hints(errors)}\n\n"
            "エラーを直した完成版のコードを、すべてのファイル（JSON を含む）についてもう一度出力してください。" + INSTRUCTIONS)


def generate(model: str, request: str, extract, max_repair: int = 3, num_ctx: int = 16384,
             lang: str = "js", compact: bool = False, examples: bool = True,
             exclude_packs: set[str] | None = None, runtime: bool = False) -> GenResult:
    """extract: 出力テキスト → (scripts, jsons) の関数（llm.eval_user.extract）。文脈は gen_context を参照。"""
    context = gen_context(request, compact, examples, exclude_packs)
    out = chat(model, gen_prompt(context, request), temperature=0.0, num_ctx=num_ctx, max_tokens=3000, timeout=1800)
    history = [out]
    scripts, jsons = extract(out, lang)
    errors = check(scripts, runtime, jsons)
    attempts = [(dict(scripts), list(jsons), errors)]
    rounds = 0
    while errors and rounds < max_repair:
        rounds += 1
        out2 = chat(model, fix_prompt(context, request, scripts, errors, jsons), temperature=0.0, num_ctx=num_ctx,
                    max_tokens=3000, timeout=1800)
        history.append(out2)
        new_scripts, new_jsons = extract(out2, lang)
        if not new_scripts and not new_jsons:
            break
        scripts = {**scripts, **new_scripts}  # 修正で出し直されなかったファイルは前回のものを残す
        jsons = new_jsons or jsons
        errors = check(scripts, runtime, jsons)
        attempts.append((dict(scripts), list(jsons), errors))
    return GenResult(out, scripts, errors, rounds, history, attempts, jsons)
