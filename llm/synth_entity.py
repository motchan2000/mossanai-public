"""「日本語の指示 → カスタムMob（entity + spawn_rules + loot_table）」の合成データ。

特徴をランダムに組み合わせ、指示文とファイル群を同時に作る。
コンポーネントの書き方は vanilla（creeper / zombie / cow 等）の実物に合わせ、
生成物は bedrock_tools で検証してエラー0件のものだけを学習に使う（build_gen_data.py）。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from bedrock_tools.jsonc import dump

ENTITY_FORMAT = "1.26.0"
SPAWN_FORMAT = "1.8.0"

# (識別子に使う英名, 日本語名)
NAMES = [
    ("ghoul", "グール"), ("wraith", "レイス"), ("slime_king", "スライムキング"), ("mud_golem", "泥ゴーレム"),
    ("night_stalker", "ナイトストーカー"), ("bomb_beetle", "ボムビートル"), ("frost_zombie", "フロストゾンビ"),
    ("ember_imp", "エンバーインプ"), ("cave_crawler", "ケイブクローラー"), ("sand_wolf", "サンドウルフ"),
    ("moss_sheep", "コケヒツジ"), ("puff_bird", "パフバード"), ("tiny_pig", "ミニブタ"), ("forest_deer", "シカ"),
    ("rock_turtle", "イワガメ"), ("glow_bug", "ヒカリムシ"), ("shadow_cat", "カゲネコ"), ("goblin", "ゴブリン"),
    ("skull_bat", "ドクロコウモリ"), ("lava_crab", "マグマガニ"), ("snow_rabbit", "ユキウサギ"),
    ("storm_spirit", "ストームスピリット"), ("bone_knight", "ボーンナイト"), ("swamp_frog", "ヌマガエル"),
]
NAMESPACES = ["my", "custom", "mod", "addon", "fantasy", "test", "mob", "adv"]

BIOMES = [  # (has_biome_tag の値, 日本語)
    ("plains", "平原"), ("forest", "森"), ("desert", "砂漠"), ("frozen", "雪原"), ("jungle", "ジャングル"),
    ("mesa", "荒野"), ("swamp", "沼地"), ("taiga", "タイガ"), ("savanna", "サバンナ"), ("mountains", "山岳"),
]
DROPS = [  # (アイテムID, 日本語)
    ("minecraft:rotten_flesh", "腐った肉"), ("minecraft:bone", "骨"), ("minecraft:gunpowder", "火薬"),
    ("minecraft:leather", "革"), ("minecraft:feather", "羽"), ("minecraft:string", "糸"),
    ("minecraft:slime_ball", "スライムボール"), ("minecraft:iron_ingot", "鉄インゴット"),
    ("minecraft:gold_nugget", "金塊"), ("minecraft:emerald", "エメラルド"), ("minecraft:beef", "生の牛肉"),
    ("minecraft:mutton", "生の羊肉"), ("minecraft:blaze_powder", "ブレイズパウダー"), ("minecraft:snowball", "雪玉"),
]
TEMPT_ITEMS = [("minecraft:wheat", "小麦"), ("minecraft:carrot", "ニンジン"), ("minecraft:apple", "リンゴ"),
               ("minecraft:sweet_berries", "スイートベリー"), ("minecraft:wheat_seeds", "種")]


DEFAULTS = {"health": 20, "speed": 0.25, "damage": 3, "explode_power": 3}


@dataclass
class Spec:
    archetype: str  # hostile | exploder | neutral | passive
    name: str
    name_ja: str
    namespace: str
    health: int
    speed: float
    damage: int = 0
    explode_power: int = 3
    burns: bool = False
    fire_immune: bool = False
    knockback: float = 0.0
    scale: float = 1.0
    swims: bool = False
    tempt: tuple[str, str] | None = None
    drop: tuple[str, str] | None = None
    drop_max: int = 2
    xp: int = 0
    spawn: dict = field(default_factory=dict)  # time: night|day|any, biome: (tag, ja)|None, underground: bool
    omitted: set = field(default_factory=set)  # 指示文で指定しない項目（出力は既定値）

    @property
    def identifier(self) -> str:
        return f"{self.namespace}:{self.name}"


def random_spec(rng: random.Random) -> Spec:
    archetype = rng.choices(["hostile", "exploder", "neutral", "passive"], weights=[4, 2, 2, 3])[0]
    name, name_ja = rng.choice(NAMES)
    hostile = archetype in ("hostile", "exploder", "neutral")
    s = Spec(
        archetype=archetype, name=name, name_ja=name_ja, namespace=rng.choice(NAMESPACES),
        health=rng.choice([4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50, 60, 80, 100]),
        speed=rng.choice([0.15, 0.2, 0.23, 0.25, 0.3, 0.35, 0.4]),
        damage=rng.choice([2, 3, 4, 5, 6, 8, 10]) if hostile else 0,
        explode_power=rng.choice([2, 3, 4, 5]),
    )
    # 指定されなかった値は既定値にする例も学習させる（v1は速度未指定で 0.025 などを出していた）
    for attr in ("health", "speed", "damage", "explode_power"):
        if rng.random() < 0.25:
            s.omitted.add(attr)
            setattr(s, attr, DEFAULTS[attr] if attr != "damage" or hostile else 0)
    s.burns = hostile and rng.random() < 0.25
    s.fire_immune = not s.burns and rng.random() < 0.15
    s.knockback = rng.choice([0.0, 0.0, 0.0, 0.5, 1.0])
    s.scale = rng.choice([1.0, 1.0, 1.0, 0.5, 0.7, 1.5, 2.0])
    s.swims = rng.random() < 0.1
    if archetype == "passive" and rng.random() < 0.6:
        s.tempt = rng.choice(TEMPT_ITEMS)
    if rng.random() < 0.7:
        s.drop = rng.choice(DROPS)
        s.drop_max = rng.choice([1, 2, 3])
    s.xp = rng.choice([0, 3, 5, 10]) if hostile else rng.choice([0, 1, 3])
    if rng.random() < 0.75:
        s.spawn = {
            "time": rng.choice(["night", "night", "day", "any"]) if hostile else rng.choice(["day", "any"]),
            "biome": rng.choice(BIOMES) if rng.random() < 0.6 else None,
            "underground": hostile and rng.random() < 0.2,
        }
    return s


# ---------------------------------------------------------------------------
# 指示文
# ---------------------------------------------------------------------------
def speed_ja(v: float) -> str:
    return "とても遅い" if v <= 0.15 else "遅い" if v < 0.23 else "普通の速さの" if v <= 0.25 else "素早い" if v <= 0.3 else "とても速い"


def instruction(s: Spec, rng: random.Random) -> str:
    parts = []
    if s.spawn:
        t = {"night": rng.choice(["夜だけ湧く", "夜に出現する", "暗い場所に湧く"]),
             "day": rng.choice(["昼に湧く", "明るい時間に出現する"]), "any": ""}[s.spawn["time"]]
        place = []
        if s.spawn.get("biome"):
            place.append(f"{s.spawn['biome'][1]}バイオームに")
        if s.spawn.get("underground"):
            place.append("地下に")
        if place or t:
            parts.append("".join(place) + (t or "自然に湧く"))
    kind = {
        "hostile": rng.choice(["プレイヤーを追いかけて攻撃する敵Mob", "プレイヤーを襲う敵対Mob", "近接攻撃してくる敵"]),
        "exploder": rng.choice(["プレイヤーに近づいて爆発する敵Mob", "クリーパーのように爆発する敵", "自爆する敵Mob"]),
        "neutral": rng.choice(["攻撃されると反撃する中立Mob", "普段はおとなしいが殴られると反撃するMob"]),
        "passive": rng.choice(["おとなしい動物Mob", "攻撃してこない友好的なMob", "のんびり歩き回る動物"]),
    }[s.archetype]
    details = []
    if "health" not in s.omitted:
        details.append(f"体力は{s.health}")
    if s.archetype == "exploder" and "explode_power" not in s.omitted:
        details.append(f"爆発の威力は{s.explode_power}")
    elif s.archetype in ("hostile", "neutral") and "damage" not in s.omitted:
        details.append(f"攻撃力は{s.damage}")
    if "speed" not in s.omitted:
        details.append(f"{speed_ja(s.speed)}移動速度（{s.speed}）")
    if s.burns:
        details.append("日光で燃える")
    if s.fire_immune:
        details.append("火や溶岩でダメージを受けない")
    if s.knockback:
        details.append("ノックバックしない" if s.knockback >= 1 else "ノックバックしにくい")
    if s.scale != 1.0:
        details.append(f"{'大きい' if s.scale > 1 else '小さい'}サイズ（{s.scale}倍）")
    if s.swims:
        details.append("水中でも呼吸できる")
    if s.tempt:
        details.append(f"{s.tempt[1]}を持っているとついてくる")
    if s.drop:
        details.append(f"倒すと{s.drop[1]}を最大{s.drop_max}個落とす")
    if s.xp:
        details.append(f"経験値を{s.xp}落とす")
    rng.shuffle(details)
    head = f"「{s.name_ja}」という" if rng.random() < 0.5 else ""
    idpart = f"IDは{s.identifier}。" if rng.random() < 0.6 else f"名前空間は{s.namespace}。"
    lead = rng.choice(["を作ってください。", "のビヘイビアを書いて。", "を追加したいです。", "を作成して。"])
    where = f"{parts[0]}、" if parts else ""
    tail = f"{'、'.join(details)}。" if details else ""
    return f"{head}{where}{kind}{lead}{idpart}{tail}"


# ---------------------------------------------------------------------------
# ファイル生成
# ---------------------------------------------------------------------------
def player_filter() -> dict:
    return {"filters": {"test": "is_family", "subject": "other", "value": "player"}, "max_dist": 16}


def entity_json(s: Spec) -> dict:
    comps: dict = {
        "minecraft:type_family": {"family": [s.name, "monster" if s.archetype in ("hostile", "exploder") else "mob"]
                                  if s.archetype != "passive" else [s.name, "animal", "mob"]},
        "minecraft:health": {"value": s.health, "max": s.health},
        "minecraft:collision_box": {"width": round(0.6 * s.scale, 2), "height": round(1.8 * s.scale, 2)
                                    if s.archetype != "passive" else round(1.0 * s.scale, 2)},
        "minecraft:movement": {"value": s.speed},
        "minecraft:movement.basic": {},
        "minecraft:navigation.walk": {"can_path_over_water": True, "avoid_damage_blocks": True},
        "minecraft:jump.static": {},
        "minecraft:can_climb": {},
        "minecraft:physics": {},
        "minecraft:pushable": {"is_pushable": True, "is_pushable_by_piston": True},
        "minecraft:nameable": {},
        "minecraft:despawn": {"despawn_from_distance": {"min_distance": 32, "max_distance": 40}},
        "minecraft:breathable": ({"total_supply": 15, "suffocate_time": 0, "breathes_water": True, "breathes_air": True}
                                 if s.swims else {"total_supply": 15, "suffocate_time": 0}),
        "minecraft:behavior.float": {"priority": 0},
        "minecraft:behavior.random_stroll": {"priority": 6, "speed_multiplier": 1.0},
        "minecraft:behavior.look_at_player": {"priority": 7, "look_distance": 6.0, "probability": 0.02},
        "minecraft:behavior.random_look_around": {"priority": 8},
    }
    groups, events = {}, {}
    if s.archetype in ("hostile", "neutral"):
        comps["minecraft:attack"] = {"damage": s.damage}
        comps["minecraft:behavior.melee_attack"] = {"priority": 3, "speed_multiplier": 1.2, "track_target": True}
        comps["minecraft:behavior.hurt_by_target"] = {"priority": 1}
    if s.archetype == "hostile":
        comps["minecraft:behavior.nearest_attackable_target"] = {
            "priority": 2, "must_see": True, "reselect_targets": True,
            "entity_types": [{"filters": {"test": "is_family", "subject": "other", "value": "player"}, "max_dist": 16}]}
    if s.archetype == "exploder":
        comps["minecraft:attack"] = {"damage": 0}
        comps["minecraft:behavior.melee_attack"] = {"priority": 4, "speed_multiplier": 1.25, "reach_multiplier": 0}
        comps["minecraft:behavior.swell"] = {"priority": 2, "start_distance": 2.5, "stop_distance": 6}
        comps["minecraft:behavior.nearest_attackable_target"] = {
            "priority": 1, "must_see": True,
            "entity_types": [{"filters": {"test": "is_family", "subject": "other", "value": "player"}}]}
        comps["minecraft:target_nearby_sensor"] = {
            "inside_range": 2.5, "outside_range": 6, "must_see": True,
            "on_inside_range": {"event": f"{s.namespace}:start_exploding", "target": "self"},
            "on_outside_range": {"event": f"{s.namespace}:stop_exploding", "target": "self"},
            "on_vision_lost_inside_range": {"event": f"{s.namespace}:stop_exploding", "target": "self"}}
        groups[f"{s.namespace}:exploding"] = {"minecraft:explode": {
            "fuse_length": 1.5, "fuse_lit": True, "power": s.explode_power, "causes_fire": False,
            "destroy_affected_by_griefing": True}}
        events[f"{s.namespace}:start_exploding"] = {"add": {"component_groups": [f"{s.namespace}:exploding"]}}
        events[f"{s.namespace}:stop_exploding"] = {"remove": {"component_groups": [f"{s.namespace}:exploding"]}}
    if s.archetype == "passive":
        comps["minecraft:behavior.panic"] = {"priority": 1, "speed_multiplier": 1.5}
        if s.tempt:
            comps["minecraft:behavior.tempt"] = {"priority": 4, "speed_multiplier": 1.2, "items": [s.tempt[0]]}
    if s.burns:
        comps["minecraft:burns_in_daylight"] = {}
    if s.fire_immune:
        comps["minecraft:fire_immune"] = {}
    if s.knockback:
        comps["minecraft:knockback_resistance"] = {"value": s.knockback}
    if s.scale != 1.0:
        comps["minecraft:scale"] = {"value": s.scale}
    if s.drop:
        comps["minecraft:loot"] = {"table": f"loot_tables/entities/{s.name}.json"}
    if s.xp:
        comps["minecraft:experience_reward"] = {"on_death": f"query.last_hit_by_player ? {s.xp} : 0"}

    body = {
        "description": {"identifier": s.identifier, "is_spawnable": True, "is_summonable": True,
                        "spawn_category": "monster" if s.archetype in ("hostile", "exploder") else "creature"},
        "components": comps,
    }
    if groups:
        body["component_groups"] = groups
    if events:
        body["events"] = events
    return {"format_version": ENTITY_FORMAT, "minecraft:entity": body}


def spawn_json(s: Spec) -> dict | None:
    if not s.spawn:
        return None
    cond: dict = {}
    if s.spawn.get("underground"):
        cond["minecraft:spawns_underground"] = {}
    else:
        cond["minecraft:spawns_on_surface"] = {}
    t = s.spawn["time"]
    if t == "night":
        cond["minecraft:brightness_filter"] = {"min": 0, "max": 7, "adjust_for_weather": True}
    elif t == "day":
        cond["minecraft:brightness_filter"] = {"min": 9, "max": 15, "adjust_for_weather": False}
    if s.archetype in ("hostile", "exploder"):
        cond["minecraft:difficulty_filter"] = {"min": "easy", "max": "hard"}
    cond["minecraft:weight"] = {"default": 40 if s.archetype != "passive" else 8}
    cond["minecraft:herd"] = {"min_size": 1, "max_size": 2 if s.archetype != "passive" else 4}
    tags = [{"test": "has_biome_tag", "value": "monster" if s.archetype in ("hostile", "exploder") else "animal"}]
    if s.spawn.get("biome"):
        tags.append({"test": "has_biome_tag", "value": s.spawn["biome"][0]})
    cond["minecraft:biome_filter"] = tags[0] if len(tags) == 1 else {"all_of": tags}
    return {"format_version": SPAWN_FORMAT, "minecraft:spawn_rules": {
        "description": {"identifier": s.identifier,
                        "population_control": "monster" if s.archetype in ("hostile", "exploder") else "animal"},
        "conditions": [cond]}}


def loot_json(s: Spec) -> dict | None:
    if not s.drop:
        return None
    entry = {"type": "item", "name": s.drop[0], "weight": 1,
             "functions": [{"function": "set_count", "count": {"min": 0, "max": s.drop_max}}]}
    if s.drop_max > 1:
        entry["functions"].append({"function": "looting_enchant", "count": {"min": 0, "max": 1}})
    return {"pools": [{"rolls": 1, "entries": [entry]}]}


def files_for(s: Spec) -> list[tuple[str, dict]]:
    out = [(f"BP/entities/{s.name}.json", entity_json(s))]
    sp = spawn_json(s)
    if sp:
        out.append((f"BP/spawn_rules/{s.name}.json", sp))
    lt = loot_json(s)
    if lt:
        out.append((f"BP/loot_tables/entities/{s.name}.json", lt))
    return out


def answer_text(files: list[tuple[str, dict]]) -> str:
    parts = ["以下のファイルを作成します。"]
    for path, doc in files:
        parts.append(f"### {path}\n```json\n{dump(doc)}\n```")
    return "\n\n".join(parts)


def sample(rng: random.Random) -> tuple[str, list[tuple[str, dict]], Spec]:
    s = random_spec(rng)
    return instruction(s, rng), files_for(s), s
