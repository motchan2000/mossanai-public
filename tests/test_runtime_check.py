"""BDS 実行テスト（遅いので BDS がある環境のみ。1件約10秒）。"""
import pytest

from bedrock_tools.runtime_check import bds_dir, classify, marks, run_pack

has_bds = (bds_dir() / "bedrock_server.exe").exists()
needs_bds = pytest.mark.skipif(not has_bds, reason="BDS 未取得（sources.json の bds）")


def test_classify_log_lines():
    log = [
        "[2026-10-02 12:28:57:491 INFO] Profiler config ('bootstrap.json') load result: success=1",
        "[2026-10-02 12:28:59:295 WARN] [Scripting] MOSSANAI_OK loaded",
        "[2026-10-02 12:29:07:349 ERROR] [Scripting] [x BP] ReferenceError: Native function [World::setDynamicProperty] cannot be used in early execution.",
        "[2026-10-02 12:29:07:400 WARN] [Json] component 'minecraft:foo' is not valid",
    ]
    errs = classify(log)
    assert len(errs) == 2 and "early execution" in errs[0] and "minecraft:foo" in errs[1]


@needs_bds
def test_valid_script_runs():
    src = ('import { world } from "@minecraft/server";\n'
           'world.afterEvents.worldLoad.subscribe(() => { console.warn("MOSSANAI_OK loaded"); });\n')
    r = run_pack({"scripts/main.js": src})
    assert r.ok, r.errors
    assert "OK loaded" in marks(r)


@needs_bds
def test_early_execution_error_is_detected():
    """型チェックでは検出できない実行時エラー。"""
    src = 'import { world } from "@minecraft/server";\nworld.setDynamicProperty("x", 1);\n'
    r = run_pack({"scripts/main.js": src})
    assert r.started and any("early execution" in e for e in r.errors)
