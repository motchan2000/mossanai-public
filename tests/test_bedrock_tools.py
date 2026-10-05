"""bedrock_tools（検証・雛形生成）のテスト。"""
import json

import pytest

from bedrock_tools.check import check_path
from bedrock_tools.jsonc import dump, parse_lenient
from bedrock_tools.manifest import check_manifest, check_pack_pair, latest_stable, new_pack_pair, scaffold
from bedrock_tools.schema_validator import SchemaValidator, default_schema_root

try:
    default_schema_root()
    HAS_SAMPLES = True
except FileNotFoundError:
    HAS_SAMPLES = False
needs_samples = pytest.mark.skipif(not HAS_SAMPLES, reason="bedrock-samples 未取得")


def test_dump_inlines_short_scalar_arrays():
    obj = {"version": [1, 0, 0], "list": [{"a": 1}], "empty": {}, "s": "日本語"}
    text = dump(obj)
    assert '"version": [1, 0, 0]' in text
    assert '"s": "日本語"' in text
    assert json.loads(text) == obj


def test_parse_lenient_reports_position():
    obj, err = parse_lenient('{\n  "a": 1\n  "b": 2\n}')
    assert obj is None and "行 3" in err


@pytest.fixture(scope="module")
def validator():
    return SchemaValidator()


ENTITY = {
    "format_version": "1.26.50",
    "minecraft:entity": {
        "description": {"identifier": "test:ghoul", "is_spawnable": True, "is_summonable": True},
        "components": {
            "minecraft:health": {"value": 20, "max": 20},
            "minecraft:movement": {"value": 0.25},
            "minecraft:behavior.melee_attack": {"priority": 3},
        },
    },
}


@needs_samples
def test_valid_entity_passes(validator):
    r = validator.validate(ENTITY)
    assert r.doc_type == "entity" and r.ok, [str(i) for i in r.issues]


@needs_samples
def test_entity_type_error_is_reported_with_path(validator):
    # behavior.melee_attack はスキーマに型定義がある
    bad = json.loads(json.dumps(ENTITY))
    bad["minecraft:entity"]["components"]["minecraft:behavior.melee_attack"]["priority"] = "high"
    r = validator.validate(bad)
    assert not r.ok
    assert any("minecraft:behavior.melee_attack/priority" in i.path for i in r.issues)


@needs_samples
def test_semantic_unknown_component_with_suggestion():
    from bedrock_tools.semantic import check_semantics

    bad = json.loads(json.dumps(ENTITY))
    bad["minecraft:entity"]["components"]["minecraft:helth"] = {"value": 10}
    errs = [i for i in check_semantics(bad) if i.severity == "error"]
    assert len(errs) == 1 and "minecraft:health" in errs[0].message


@needs_samples
def test_semantic_dynamic_component_type_warning():
    """スキーマが Dynamic value の movement も、vanilla由来のプロファイルで型違いを警告。"""
    from bedrock_tools.semantic import check_semantics

    bad = json.loads(json.dumps(ENTITY))
    bad["minecraft:entity"]["components"]["minecraft:movement"]["value"] = "fast"
    warns = [i for i in check_semantics(bad) if i.severity == "warning"]
    assert any(i.path.endswith("minecraft:movement/value") and "number" in i.message for i in warns)


@needs_samples
def test_semantic_event_references_missing_group():
    from bedrock_tools.semantic import check_semantics

    doc = json.loads(json.dumps(ENTITY))
    doc["minecraft:entity"]["component_groups"] = {"test:angry": {"minecraft:angry": {"duration": 10}}}
    doc["minecraft:entity"]["events"] = {
        "test:become_angry": {"add": {"component_groups": ["test:angy"]}},
        "test:calm": {"sequence": [{"remove": {"component_groups": ["test:angry"]}}]},
    }
    found = [i for i in check_semantics(doc) if "component_group" in i.message]
    assert len(found) == 1 and "test:angy" in found[0].message and "test:angry" in found[0].message
    assert found[0].severity == "warning"


@needs_samples
def test_spawn_rules_unknown_keys_are_errors():
    """mossanai-v2 が実際に出力した架空のキー（weight_overrides）を検出できること。"""
    from bedrock_tools.semantic import check_semantics

    doc = {"format_version": "1.8.0", "minecraft:spawn_rules": {
        "description": {"identifier": "test:x", "population_control": "monster"},
        "conditions": [{"minecraft:spawns_on_surface": {}, "minecraft:brightnes_filter": {"min": 0, "max": 7}}],
        "weight_overrides": [{"biome_tags": ["desert"], "overwrites": 4}]}}
    errs = [i for i in check_semantics(doc) if i.severity == "error"]
    assert any("weight_overrides" in i.path for i in errs)
    assert any("minecraft:brightness_filter" in i.message for i in errs)


@needs_samples
def test_vanilla_spawn_rules_have_no_semantic_errors():
    from pathlib import Path

    from bedrock_tools.semantic import check_semantics

    src = sorted(Path("data/bedrock/raw").glob("bedrock-samples-*"))[-1] / "behavior_pack" / "spawn_rules"
    bad = {}
    for p in src.glob("*.json"):
        doc, _ = parse_lenient(p.read_text(encoding="utf-8"))
        errs = [str(i) for i in check_semantics(doc) if i.severity == "error"]
        if errs:
            bad[p.name] = errs[:2]
    assert not bad, bad


@needs_samples
def test_vanilla_entities_have_no_semantic_errors():
    """誤検出チェック: vanillaのentityで意味エラーが出ないこと。"""
    from pathlib import Path

    from bedrock_tools.semantic import check_semantics

    src = sorted(Path("data/bedrock/raw").glob("bedrock-samples-*"))[-1] / "behavior_pack" / "entities"
    bad = {}
    for p in src.glob("*.json"):
        doc, _ = parse_lenient(p.read_text(encoding="utf-8"))
        errs = [str(i) for i in check_semantics(doc) if i.severity == "error"]
        if errs:
            bad[p.name] = errs[:2]
    assert not bad, bad


@needs_samples
def test_schema_version_selection(validator):
    from bedrock_tools.schema_validator import DOC_TYPES

    entity = next(t for t in DOC_TYPES if t.name == "entity")
    assert "/1.26.50/" in validator.pick_schema(entity, "1.26.50")
    assert "/1.21.80/" in validator.pick_schema(entity, "1.10.0")  # 古すぎる場合は最古
    assert "/1.21.90/" in validator.pick_schema(entity, "1.21.95")


@needs_samples
def test_legacy_item_is_flagged(validator):
    r = validator.validate({"format_version": "1.10", "minecraft:item": {
        "description": {"identifier": "test:x"}, "components": {"minecraft:food": {"saturation_modifier": "low"}}}})
    assert r.legacy


@needs_samples
def test_old_block_format_is_not_judged_by_new_schema(validator):
    """ブロックのスキーマは 1.26.20 版のみ。1.21.x の数値ステートを誤ってエラーにしない（ユーザーの実ファイルで発覚）。"""
    from bedrock_tools.check import check_file
    from bedrock_tools.jsonc import dump

    doc = {"format_version": "1.21.20", "minecraft:block": {
        "description": {"identifier": "test:crop", "states": {"test:growth": [0, 1, 2, 3]}},
        "components": {"minecraft:custom_components": ["test:grow"],
                       "minecraft:material_instances": {"*": {"texture": "crop", "render_method": "alpha_test",
                                                              "ambient_occlusion": False}}}}}
    r = validator.validate(doc)
    assert r.legacy
    import tempfile
    from pathlib import Path

    p = Path(tempfile.mkdtemp()) / "crop.json"
    p.write_text(dump(doc), encoding="utf-8")
    rep = check_file(p, validator)
    assert rep.errors == [], rep.errors
    assert any("旧方式" in w for w in rep.warnings)


def test_generated_pack_pair_is_clean():
    bp, rp = new_pack_pair("Test", script_entry="scripts/main.js")
    assert check_manifest(bp) == [] and check_manifest(rp) == []
    assert check_pack_pair(bp, rp) == []
    uuids = [bp["header"]["uuid"], rp["header"]["uuid"]] + [m["uuid"] for m in bp["modules"] + rp["modules"]]
    assert len(set(uuids)) == len(uuids)


def test_manifest_errors():
    bp, _ = new_pack_pair("Test", script_entry="scripts/main.js")
    bp["modules"][0]["uuid"] = bp["header"]["uuid"]  # 重複
    bp["modules"][1]["entry"] = "main.ts"
    bp["dependencies"][-1]["version"] = "9.9.9"
    msgs = " ".join(i.message for i in check_manifest(bp) if i.severity == "error")
    assert "重複" in msgs and "entry" in msgs
    if latest_stable("@minecraft/server"):
        assert "存在しません" in msgs


@needs_samples
def test_scaffold_then_check(tmp_path):
    scaffold(tmp_path, "My Addon", "desc", with_script=True)
    report = check_path(tmp_path)
    assert report.error_count == 0 and report.warning_count == 0
    assert (tmp_path / "My_Addon_BP" / "scripts" / "main.js").exists()


@needs_samples
def test_legacy_block_still_errors_on_fatal_mistakes(validator, tmp_path):
    """旧形式でも、ゲームが読み込めない誤り（BDS 1.26 で確認）はエラーのまま。
    ブロックの未知コンポーネント（旧名 destroy_time 等）と、custom_components にオブジェクトを書く誤り。"""
    from bedrock_tools.check import check_file
    from bedrock_tools.jsonc import dump

    doc = {"format_version": "1.20.0", "minecraft:block": {
        "description": {"identifier": "myrest:stove"},
        "components": {"minecraft:destroy_time": 3, "minecraft:custom_components": [{"name": "myrest:stove"}]}}}
    p = tmp_path / "stove.json"
    p.write_text(dump(doc), encoding="utf-8")
    errors = check_file(p, validator).errors
    assert any("destroy_time" in e for e in errors), errors
    assert any("custom_components" in e for e in errors), errors


@needs_samples
def test_legacy_item_unknown_component_is_only_warning(validator, tmp_path):
    """アイテムの未知コンポーネントはゲームでは無視されるだけなので、旧形式では警告。"""
    from bedrock_tools.check import check_file
    from bedrock_tools.jsonc import dump

    doc = {"format_version": "1.16.100", "minecraft:item": {
        "description": {"identifier": "test:x"}, "components": {"minecraft:foil": True}}}
    p = tmp_path / "x.json"
    p.write_text(dump(doc), encoding="utf-8")
    rep = check_file(p, validator)
    assert rep.errors == [], rep.errors
