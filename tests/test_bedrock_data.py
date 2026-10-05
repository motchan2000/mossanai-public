"""STEP2 / STEP9: コーパス生成・API変換・トークナイザのテスト。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "bedrock"))

from build_corpus import dump_json as dump, normalize_json, split_large_json, strip_json_comments, tag_for  # noqa: E402
from script_api import ts_type  # noqa: E402
from train_tokenizer import split_documents, train  # noqa: E402


def test_strip_comments_keeps_slashes_in_strings():
    src = '{"path": "textures/entity//pig", // comment\n "a": 1 /* block */}'
    assert json.loads(strip_json_comments(src)) == {"path": "textures/entity//pig", "a": 1}


def test_normalize_json_handles_trailing_comma_and_bom():
    body, ok = normalize_json('﻿{"a": [1, 2,],}')
    assert ok and json.loads(body) == {"a": [1, 2]}
    assert body == '{\n  "a": [1, 2]\n}'


def test_split_large_json_keeps_valid_structure():
    obj = {"format_version": "1.20.20",
           "sound_definitions": {f"mob.s{i}": {"sounds": ["x" * 50]} for i in range(100)}}
    chunks = split_large_json(obj, max_chars=1000)
    assert len(chunks) > 1
    merged = {}
    for c in chunks:
        assert c["format_version"] == "1.20.20"
        assert len(dump(c)) <= 1200
        merged.update(c["sound_definitions"])
    assert merged == obj["sound_definitions"]


def test_split_large_json_top_level_collection_does_not_blow_up():
    """blocks.json型（トップレベル自体が巨大な辞書）でも、分割後の総量が元とほぼ同じ。"""
    obj = {f"block_{i}": {"sound": "wood", "textures": {"up": f"t{i}", "down": f"t{i}"}} for i in range(500)}
    chunks = split_large_json(obj, max_chars=2000)
    total = sum(len(json.dumps(c, indent=2)) for c in chunks)
    assert total < len(json.dumps(obj, indent=2)) * 1.1
    merged = {}
    for c in chunks:
        merged.update(c)
    assert merged == obj


def test_split_large_json_does_not_duplicate_big_siblings():
    obj = {"format_version": "1",
           "a": {f"k{i}": "x" * 40 for i in range(300)},
           "b": {f"k{i}": "y" * 40 for i in range(100)}}
    chunks = split_large_json(obj, max_chars=2000)
    total = sum(len(json.dumps(c, indent=2)) for c in chunks)
    assert total < len(json.dumps(obj, indent=2)) * 1.2


def test_tag_for_paths():
    assert tag_for("behavior_pack/entities/zombie.json") == "BP_ENTITY"
    assert tag_for("behavior_pack/manifest.json") == "MANIFEST"
    assert tag_for("resource_pack/animation_controllers/pig.json") == "RP_ANIM_CTRL"
    assert tag_for("resource_pack/ui/hud_screen.json") is None


def test_ts_type_rendering():
    opt = {"name": "optional", "optional_type": {"name": "Effect"}}
    assert ts_type(opt) == "Effect | undefined"
    arr = {"name": "array", "element_type": {"name": "variant", "variant_types": [{"name": "string"}, {"name": "int32"}]}}
    assert ts_type(arr) == "(string | number)[]"
    clo = {"name": "closure", "closure_type": {"argument_types": [{"name": "EntitySpawnAfterEvent"}],
                                                "return_type": {"name": "undefined"}}}
    assert ts_type(clo) == "(arg0: EntitySpawnAfterEvent) => void"


def test_split_documents_spans_newlines():
    text = '<BP_ENTITY>{\n  "a": 1\n}</BP_ENTITY>\n<DOC># x\ny</DOC>\n'
    assert split_documents(text) == ['<BP_ENTITY>{\n  "a": 1\n}</BP_ENTITY>\n', "<DOC># x\ny</DOC>\n"]


@pytest.fixture(scope="module")
def tiny_domain_tokenizer(tmp_path_factory):
    entity = json.dumps({"format_version": "1.21.0", "minecraft:entity": {"components": {
        "minecraft:health": {"value": 20}, "minecraft:behavior.melee_attack": {"priority": 3}}}}, indent=2)
    path = tmp_path_factory.mktemp("tok") / "corpus.txt"
    path.write_text(f"<BP_ENTITY>{entity}</BP_ENTITY>\n" * 200, encoding="utf-8")
    return train("domain", [str(path)], vocab_size=800)


def test_domain_tokenizer_roundtrip_and_tags(tiny_domain_tokenizer):
    tok = tiny_domain_tokenizer
    text = '<BP_ENTITY>{\n  "minecraft:health": {"value": 7}, "名前": "ゾンビ"}</BP_ENTITY>\n'
    ids = tok.encode(text).ids
    assert tok.decode(ids, skip_special_tokens=False) == text
    assert ids[0] == tok.token_to_id("<BP_ENTITY>")


def test_domain_tokenizer_learns_whole_keys(tiny_domain_tokenizer):
    assert len(tiny_domain_tokenizer.encode('"minecraft:health"').ids) == 1
