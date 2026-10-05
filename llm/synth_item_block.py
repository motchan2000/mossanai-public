"""「日本語の指示 → カスタムアイテム / ブロック」の合成データ。

書き方は vanilla の 1.21.80+ 形式のアイテム・ブロック定義に合わせる。
"""
from __future__ import annotations

import random

from llm.synth_entity import NAMESPACES

ITEM_FORMAT = "1.21.80"
BLOCK_FORMAT = "1.21.80"

FOODS = [("magic_berry", "マジックベリー"), ("star_bread", "スターブレッド"), ("rice_ball", "おにぎり"),
         ("honey_cake", "ハニーケーキ"), ("glow_apple", "光るリンゴ"), ("dragon_meat", "ドラゴン肉")]
WEAPONS = [("ruby_sword", "ルビーの剣"), ("bone_club", "骨のこん棒"), ("crystal_dagger", "クリスタルの短剣"),
           ("thunder_blade", "雷の剣"), ("obsidian_axe", "黒曜石の斧")]
MATERIALS = [("ruby", "ルビー"), ("magic_dust", "魔法の粉"), ("ancient_coin", "古代のコイン"),
             ("dragon_scale", "ドラゴンの鱗"), ("sapphire", "サファイア")]
BLOCKS = [("ruby_block", "ルビーブロック"), ("glow_stone_brick", "光る石レンガ"), ("marble", "大理石"),
          ("ice_brick", "氷レンガ"), ("lucky_block", "ラッキーブロック"), ("moss_tile", "苔タイル"),
          ("crystal_ore", "クリスタル鉱石"), ("bounce_pad", "バウンドパッド")]
EFFECTS = [("speed", "移動速度上昇"), ("regeneration", "再生能力"), ("night_vision", "暗視"),
           ("jump_boost", "跳躍力上昇"), ("strength", "攻撃力上昇"), ("fire_resistance", "火炎耐性")]
CATEGORIES = {"food": "items", "weapon": "equipment", "material": "items", "block": "construction"}


def item_sample(rng: random.Random) -> tuple[str, list[tuple[str, dict]]]:
    kind = rng.choice(["food", "weapon", "material"])
    ns = rng.choice(NAMESPACES)
    name, ja = rng.choice({"food": FOODS, "weapon": WEAPONS, "material": MATERIALS}[kind])
    ident = f"{ns}:{name}"
    comps: dict = {"minecraft:icon": name, "minecraft:display_name": {"value": ja}}
    details = []
    stack = 64
    if kind == "food":
        nutrition = rng.choice([2, 3, 4, 6, 8, 10])
        sat = rng.choice([0.3, 0.6, 1.2])
        comps["minecraft:food"] = {"nutrition": nutrition, "saturation_modifier": sat,
                                   "can_always_eat": rng.random() < 0.3}
        comps["minecraft:use_animation"] = "eat"
        comps["minecraft:use_modifiers"] = {"use_duration": 1.6, "movement_modifier": 0.35}
        details.append(f"満腹度の回復量は{nutrition}、隠し満腹度の倍率は{sat}")
        if comps["minecraft:food"]["can_always_eat"]:
            details.append("お腹がいっぱいでも食べられる")
    elif kind == "weapon":
        dmg = rng.choice([5, 6, 7, 8, 10, 12])
        dur = rng.choice([250, 500, 1000, 1561, 2031])
        stack = 1
        comps["minecraft:damage"] = {"value": dmg}
        comps["minecraft:durability"] = {"max_durability": dur}
        comps["minecraft:hand_equipped"] = {"value": True}
        comps["minecraft:enchantable"] = {"slot": "sword", "value": rng.choice([10, 14, 22])}
        details.append(f"攻撃力は{dmg}、耐久値は{dur}")
        details.append("エンチャントできる")
    else:
        if rng.random() < 0.3:
            comps["minecraft:glint"] = True
            details.append("エンチャントされたように光る")
        if rng.random() < 0.3:
            stack = 16
    comps["minecraft:max_stack_size"] = stack
    if stack != 64 or kind == "material":
        details.append(f"最大スタック数は{stack}")
    doc = {"format_version": ITEM_FORMAT, "minecraft:item": {
        "description": {"identifier": ident, "menu_category": {"category": CATEGORIES[kind]}},
        "components": comps}}
    noun = {"food": "食べ物", "weapon": "武器", "material": "素材アイテム"}[kind]
    ins = (f"「{ja}」という{noun}を作ってください。IDは{ident}。" + "、".join(details) + "。"
           + ("アイコンのテクスチャ名は" + name + "にしてください。" if rng.random() < 0.5 else ""))
    return ins, [(f"BP/items/{name}.json", doc)]


def block_sample(rng: random.Random) -> tuple[str, list[tuple[str, dict]]]:
    ns = rng.choice(NAMESPACES)
    name, ja = rng.choice(BLOCKS)
    ident = f"{ns}:{name}"
    hard = rng.choice([0.5, 1.0, 1.5, 3.0, 5.0, 10.0])
    resist = rng.choice([1, 3, 6, 15, 100, 1200])
    comps: dict = {
        "minecraft:destructible_by_mining": {"seconds_to_destroy": hard},
        "minecraft:destructible_by_explosion": {"explosion_resistance": resist},
        "minecraft:geometry": "minecraft:geometry.full_block",
        "minecraft:material_instances": {"*": {"texture": name, "render_method": "opaque"}},
        "minecraft:map_color": rng.choice(["#9A3B3B", "#E0E0E0", "#7FB238", "#A0A0FF", "#FFD700"]),
    }
    details = [f"壊すのにかかる時間は{hard}秒", f"爆発耐性は{resist}"]
    if rng.random() < 0.4:
        light = rng.choice([7, 10, 12, 15])
        comps["minecraft:light_emission"] = light
        details.append(f"明るさ{light}で光る")
    if rng.random() < 0.3:
        fr = rng.choice([0.1, 0.4, 0.8])
        comps["minecraft:friction"] = fr
        details.append(f"摩擦は{fr}")
    if rng.random() < 0.4:
        comps["minecraft:flammable"] = {"catch_chance_modifier": 5, "destroy_chance_modifier": 20}
        details.append("燃える")
    if rng.random() < 0.5:
        comps["minecraft:loot"] = f"loot_tables/blocks/{name}.json"
        details.append(f"ルートテーブルは loot_tables/blocks/{name}.json")
    doc = {"format_version": BLOCK_FORMAT, "minecraft:block": {
        "description": {"identifier": ident, "menu_category": {"category": "construction"}},
        "components": comps}}
    ins = f"「{ja}」というブロックを追加して。IDは{ident}。" + "、".join(details) + f"。テクスチャ名は{name}。"
    return ins, [(f"BP/blocks/{name}.json", doc)]
