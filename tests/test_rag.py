"""Script API 用 RAG（索引・検索・修正ヒント）のテスト。"""
import pytest

from bedrock_tools.script_check import available

pytestmark = pytest.mark.skipif(not available(), reason="script_check 未セットアップ（npm install）")


def test_index_has_core_declarations():
    from bedrock_tools.api_index import load_index, signature

    idx = load_index()
    for name in ("World", "Player", "ActionFormData", "ModalFormData", "CustomCommandRegistry", "StartupEvent"):
        assert name in idx
    assert idx["ActionFormData"].module == "@minecraft/server-ui"
    assert "setActionBar(" in signature("ScreenDisplay")


def test_event_table_lists_2x_events_only():
    from bedrock_tools.api_index import event_table

    table = event_table()
    assert "system.beforeEvents.startup" in table
    assert "world.afterEvents.worldLoad" in table
    assert "worldInitialize" not in table


@pytest.mark.parametrize("text,expected", [
    ("/menu っていうカスタムコマンドを作って ActionFormData でメニュー", {"CustomCommandRegistry", "ActionFormData"}),
    ("スコアボード money を作ってサイドバーに表示して", {"ScoreboardObjective", "DisplaySlotId"}),
    ("剣でモブを倒したらloreに討伐数", {"EntityDieAfterEvent", "ItemStack"}),
    ("回復の杖のカスタムコンポーネント", {"ItemComponentRegistry", "ItemCustomComponent"}),
])
def test_retrieve(text, expected):
    from mcaddon.rag import retrieve

    assert expected <= set(retrieve(text))


def test_repair_hints_point_to_right_place():
    from mcaddon.rag import repair_hints

    hints = repair_hints([
        "scripts/main.js:4:7 TS2339 Property 'getScore' does not exist on type 'Player'.",
        "scripts/main.js:1:33 TS2305 Module '\"@minecraft/server\"' has no exported member 'ModalFormData'.",
    ])
    assert "ScoreboardObjective" in hints  # getScore の本当の持ち主
    assert "@minecraft/server-ui" in hints  # ModalFormData の import 元


def test_pattern_hints():
    from mcaddon.rag import repair_hints

    hints = repair_hints([
        "a.js:1:1 TS2345 Argument of type 'Entity' is not assignable to parameter of type 'Player'.",
        "a.js:2:1 TS2322 Type '\"String\"' is not assignable to type 'CustomCommandParamType'.",
        "a.js:3:1 TS2345 Argument of type 'number' is not assignable to parameter of type 'ModalFormDataDropdownOptions'.",
    ])
    assert "instanceof Player" in hints
    assert "CustomCommandParamType.String" in hints
    assert "defaultValueIndex" in hints


def test_hint_for_early_execution():
    from mcaddon.rag import repair_hints

    hints = repair_hints(["実行時エラー: [Scripting] ReferenceError: Native property getter [World::scoreboard] "
                          "cannot be used in early execution. at <anonymous> (main.js:5)"])
    assert "worldLoad" in hints


def test_hint_for_missing_project_module():
    from mcaddon.rag import repair_hints

    hints = repair_hints(["a.js:2:20 TS2307 Cannot find module '../company/Company.js' or its corresponding type declarations."])
    assert "1つのファイルで完結" in hints


def test_examples_exclude_pack():
    from mcaddon.examples import find_examples, load_examples

    if not load_examples()[0]:
        pytest.skip("data/user 未作成")
    ex = find_examples("マナポーションを飲み終わったらマナを回復", k=5, exclude_packs={"newNSPVE"})
    assert all(e.pack != "newNSPVE" for e in ex)
