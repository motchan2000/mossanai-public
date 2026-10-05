"""Bedrock Dedicated Server（BDS）でアドオンを実際に読み込ませ、実行時のエラーを確認する。

    from bedrock_tools.runtime_check import run_pack
    result = run_pack({"scripts/main.js": "..."}, json_files={"entities/x.json": "..."}, commands=["scriptevent my:test"])

型チェックでは分からない「実行時エラー」「権限エラー（early-execution / beforeEvents）」「コンテンツログのエラー」を検出する。
テスト専用の BDS（sources.json の bds, tools/bds/...）とテスト専用ワールドを使い、毎回ワールドを作り直す。
ポートは通常の 19132 とぶつからないよう 19222 / 19223 を使う。
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from bedrock_tools.jsonc import dump
from bedrock_tools.manifest import new_pack_pair
from bedrock_tools.schema_validator import PROJECT_ROOT

LEVEL = "mossanai_test"
PORT, PORT_V6 = 19222, 19223
# BDS のログ行: "[2026-10-02 12:29:07:349 ERROR] [Scripting] ..."
LEVEL_RE = re.compile(r"^\[[\d\- :]+ (?P<level>[A-Z]+)\]\s*(?P<msg>.*)$")
CONTENT_TAG_RE = re.compile(r"^\[(Scripting|Json|Molang|Blocks|Item|Actor|Recipes|Animation|Commands|Pack|"
                            r"Entity|Texture|Geometry|Sound|Loot|Spawning)[^\]]*\]", re.I)
IGNORE_RE = re.compile(r"telemetry|No CDN config|NO LOG FILE|Content logging to console is enabled|"
                       r"Unable to find the device id|Profiler config", re.I)
MARK = "MOSSANAI_"  # テスト用スクリプトが console.warn で出す目印（MOSSANAI_OK / MOSSANAI_FAIL など）


def bds_dir() -> Path:
    src = json.loads((PROJECT_ROOT / "sources.json").read_text(encoding="utf-8"))["bds"]
    return PROJECT_ROOT / src["local_dir"]


@dataclass
class RuntimeResult:
    started: bool
    errors: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.started and not self.errors


def _write_properties(server: Path) -> None:
    props = (server / "server.properties").read_text(encoding="utf-8").splitlines()
    want = {"level-name": LEVEL, "server-port": str(PORT), "server-portv6": str(PORT_V6), "allow-cheats": "true",
            "online-mode": "false", "allow-list": "false", "enable-lan-visibility": "false",
            "content-log-console-output-enabled": "true", "content-log-file-enabled": "false",
            "content-log-level": "warning", "view-distance": "5", "tick-distance": "4", "max-players": "1",
            "level-seed": "mossanai"}
    out, seen = [], set()
    for line in props:
        key = line.split("=", 1)[0]
        if key in want:
            out.append(f"{key}={want[key]}")
            seen.add(key)
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in want.items() if k not in seen]
    (server / "server.properties").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


def _entry_script(scripts: dict[str, str]) -> tuple[dict[str, str], str]:
    """エントリポイントを決める。main.js / index.js があればそれ、無ければ全ファイルを import する main を作る。"""
    for cand in ("scripts/main.js", "scripts/index.js"):
        if cand in scripts:
            return scripts, cand
    if len(scripts) == 1:
        return scripts, next(iter(scripts))
    entry = "scripts/__entry.js"
    imports = "\n".join(f'import "./{Path(p).relative_to("scripts").as_posix()}";' for p in scripts)
    return {**scripts, entry: imports + "\n"}, entry


def pack_json_files(bodies: list[str]) -> dict[str, str]:
    """生成された JSON 本文のうちブロック・アイテム・エンティティを、BP 内のパス付きで返す
    （スクリプトのカスタムコンポーネントが JSON から使われているかもゲームが確認するため、一緒に読み込ませる）。"""
    from bedrock_tools.jsonc import parse_lenient

    out = {}
    for i, body in enumerate(bodies):
        doc, err = parse_lenient(body)
        if err or not isinstance(doc, dict):
            continue
        folder = next((f for k, f in (("minecraft:block", "blocks"), ("minecraft:item", "items"),
                                      ("minecraft:entity", "entities")) if k in doc), None)
        if folder:
            out[f"{folder}/gen_{i}.json"] = body
    return out


def _module_versions(scripts: dict[str, str]) -> dict[str, str]:
    text = "\n".join(scripts.values())
    mods = {"@minecraft/server": "2.10.0"}
    if "@minecraft/server-ui" in text:
        mods["@minecraft/server-ui"] = "2.2.0"
    return mods


def run_pack(scripts: dict[str, str], json_files: dict[str, str] | None = None,
             commands: list[str] | None = None, wait_after: float = 5.0, timeout: float = 90.0) -> RuntimeResult:
    """scripts: {"scripts/main.js": ...}、json_files: {"entities/x.json": ...}（BP 内の相対パス）。"""
    server = bds_dir()
    if not (server / "bedrock_server.exe").exists():
        raise RuntimeError(f"BDS が見つかりません: {server}（sources.json の bds を参照）")
    _write_properties(server)
    world = server / "worlds" / LEVEL
    shutil.rmtree(world, ignore_errors=True)
    bp_dir = world / "behavior_packs" / "mossanai_test_bp"
    bp_dir.mkdir(parents=True)

    files, entry = _entry_script(scripts) if scripts else ({}, None)
    bp, _ = new_pack_pair("mossanai test", script_entry=entry, script_modules=_module_versions(scripts) if scripts else None)
    bp["dependencies"] = [d for d in bp["dependencies"] if "module_name" in d]  # RP は使わない
    (bp_dir / "manifest.json").write_text(dump(bp), encoding="utf-8")
    for rel, text in {**files, **(json_files or {})}.items():
        p = bp_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    (world / "world_behavior_packs.json").write_text(
        json.dumps([{"pack_id": bp["header"]["uuid"], "version": bp["header"]["version"]}]), encoding="utf-8")

    t0 = time.time()
    proc = subprocess.Popen([str(server / "bedrock_server.exe")], cwd=server, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1)
    log: list[str] = []
    started = threading.Event()

    def reader():
        for line in proc.stdout:
            line = line.rstrip()
            log.append(line)
            if "Server started" in line:
                started.set()

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    try:
        started.wait(timeout)
        if started.is_set():
            time.sleep(2.0)  # worldLoad 後の処理を待つ
            for cmd in commands or []:
                proc.stdin.write(cmd + "\n")
                proc.stdin.flush()
                time.sleep(1.0)
            time.sleep(wait_after)
        try:
            proc.stdin.write("stop\n")
            proc.stdin.flush()
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    finally:
        if proc.poll() is None:
            proc.kill()
        th.join(timeout=5)
    return RuntimeResult(started.is_set(), classify(log), log, time.time() - t0)


def classify(log: list[str]) -> list[str]:
    """エラーとみなす行: ERROR レベル、またはコンテンツログ（[Scripting] 等）の WARN のうち目印以外。"""
    errors = []
    for line in log:
        m = LEVEL_RE.match(line)
        if not m or IGNORE_RE.search(line):
            continue
        level, msg = m["level"], m["msg"]
        if MARK in msg and "FAIL" not in msg:
            continue
        if level == "ERROR" or (level == "WARN" and CONTENT_TAG_RE.match(msg)) or (MARK in msg and "FAIL" in msg):
            errors.append(msg)
    return errors


def marks(result: RuntimeResult) -> list[str]:
    """テスト用スクリプトが出した目印（MOSSANAI_xxx）の一覧。"""
    return [l.split(MARK, 1)[1] for l in result.log if MARK in l]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    good = ('import { world, system } from "@minecraft/server";\n'
            'world.afterEvents.worldLoad.subscribe(() => { console.warn("MOSSANAI_OK loaded"); });\n')
    r = run_pack({"scripts/main.js": good})
    print("good:", r.started, r.errors, f"{r.seconds:.1f}s")
    print("\n".join(l for l in r.log if "MOSSANAI" in l or "Script" in l)[:800])
    bad = ('import { world } from "@minecraft/server";\n'
           'world.setDynamicProperty("x", 1);  // early-execution では world の変更不可\n')
    r = run_pack({"scripts/main.js": bad})
    print("bad:", r.started, r.errors[:3], f"{r.seconds:.1f}s")
