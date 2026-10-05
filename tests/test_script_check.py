"""Script API 型チェックのテスト（@minecraft/server 2.10.0）。"""
import pytest

from bedrock_tools.script_check import available, check_scripts

pytestmark = pytest.mark.skipif(not available(), reason="script_check 未セットアップ（npm install）")

GOOD = """import { world, system } from "@minecraft/server";

world.afterEvents.playerSpawn.subscribe((ev) => {
  if (ev.initialSpawn) {
    ev.player.sendMessage("ようこそ");
  }
});

system.runInterval(() => {
  for (const p of world.getAllPlayers()) {
    p.addEffect("speed", 40, { amplifier: 1, showParticles: false });
  }
}, 20);
"""


def test_valid_js_passes():
    assert check_scripts({"scripts/main.js": GOOD}) == []


def test_removed_1x_api_is_detected():
    """2.0 で削除された worldInitialize（1.x 時代のコード）を検出する。"""
    src = 'import { world } from "@minecraft/server";\nworld.afterEvents.worldInitialize.subscribe(() => {});\n'
    issues = check_scripts({"scripts/main.js": src})
    assert len(issues) == 1 and "worldInitialize" in issues[0].message and issues[0].line == 2


def test_wrong_argument_type_and_typo():
    src = ('import { world } from "@minecraft/server";\n'
           'world.sendMesage("hi");\n'
           'world.getDimension("overworld").spawnEntity("minecraft:zombie", 1);\n')
    issues = check_scripts({"scripts/main.js": src})
    assert {i.line for i in issues} == {2, 3}
    assert any("sendMessage" in i.message for i in issues)  # "Did you mean" 候補


def test_typescript_and_multiple_files():
    files = {
        "scripts/util.ts": 'import { Player } from "@minecraft/server";\n'
                           "export function greet(p: Player): void { p.sendMessage(`hi ${p.name}`); }\n",
        "scripts/main.ts": 'import { world } from "@minecraft/server";\nimport { greet } from "./util";\n'
                           "world.afterEvents.playerSpawn.subscribe((e) => greet(e.player));\n",
    }
    assert check_scripts(files) == []


def test_console_is_available():
    """Minecraft のスクリプト環境では console.warn 等が使える。"""
    assert check_scripts({"scripts/main.js": 'console.warn("debug");\n'}) == []


def test_unknown_module():
    issues = check_scripts({"scripts/main.js": 'import { world } from "@minecraft/server-net";\n'})
    assert issues and issues[0].code == "TS2307"


def test_custom_block_state_is_not_an_error_in_js():
    """getState / withState はバニラのステート名に型付けされているが、JS ではカスタムステートも動く（BDS で確認）。"""
    code = ('import { system } from "@minecraft/server";\n'
            'system.beforeEvents.startup.subscribe((ev) => {\n'
            '  ev.blockComponentRegistry.registerCustomComponent("my:lamp", {\n'
            '    onPlayerInteract(e) {\n'
            '      const lit = e.block.permutation.getState("my:lit");\n'
            '      e.block.setPermutation(e.block.permutation.withState("my:lit", !lit));\n'
            '    },\n'
            '  });\n'
            '});\n')
    assert check_scripts({"scripts/main.js": code}) == []
    assert check_scripts({"scripts/main.ts": code})  # TS では型の指定が必要なので残す
