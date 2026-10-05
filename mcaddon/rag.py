"""Script API 生成のための RAG（公式型定義を参照しながら書かせる）。

  1. 依頼文の日本語キーワードから関係する API（イベント・クラス）を選ぶ
  2. 早見表（2.x の注意点）+ イベント一覧 + 選んだ宣言をプロンプトに入れる
  3. tsc の型チェックエラーを、該当クラスの宣言やメンバーの所在ヒント付きで返して直させる
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

from bedrock_tools.api_index import event_payload, event_table, load_index, signature

# 2.x の正しい書き方と、よくある間違い（公式型定義 2.10.0 で確認済みの内容のみ）
CHEATSHEET = """\
# @minecraft/server 2.x（安定版）の要点
- import は `@minecraft/server`（world, system, ItemStack など）。フォーム（ActionFormData / ModalFormData / MessageFormData）は `@minecraft/server-ui` から。
- 起動時の登録は system.beforeEvents.startup.subscribe((ev) => { ... }) の中で行う:
  - カスタムコマンド: ev.customCommandRegistry.registerCommand({ name: "ns:cmd", description, permissionLevel: CommandPermissionLevel.Any, mandatoryParameters: [{ name, type: CustomCommandParamType.Integer }] }, (origin, ...args) => { ...; return { status: CustomCommandStatus.Success }; })
    コマンド名は "名前空間:名前"。パラメータ型は列挙型（CustomCommandParamType.String など、文字列 "String" は不可）。
- origin.sourceEntity / ev.sourceEntity / damageSource.damagingEntity などは Entity 型。Player のメソッド（sendMessage, onScreenDisplay, selectedSlotIndex, フォームの show など）を使う前に `if (!(e instanceof Player)) return;` で確認する。
- ModalFormData の初期値はオプションのオブジェクトで渡す: dropdown(label, items, { defaultValueIndex: 0 }) / slider(label, min, max, { defaultValue, valueStep }) / toggle(label, { defaultValue }) / textField(label, placeholder, { defaultValue })。
  - カスタムコンポーネント: ev.itemComponentRegistry.registerCustomComponent("ns:name", { onUse(e) {...} }) / ev.blockComponentRegistry.registerCustomComponent("ns:name", { onPlayerInteract(e) {...} })
  - カスタムアイテム・ブロックの JSON: format_version "1.21.90"。components に "ns:name": {} と書くとスクリプトのカスタムコンポーネントと紐付く（minecraft:custom_components は旧方式で不可）。ブロックの破壊時間は minecraft:destructible_by_mining { seconds_to_destroy }、明るさは minecraft:light_emission、onTick には minecraft:tick { interval_range: [20, 20], looping: true } が必要。ステートは description.states { "ns:lit": [false, true] }、見た目の切り替えは permutations の condition "q.block_state('ns:lit')"。
  - ブロックのステート: block.permutation.getState("ns:lit") / block.setPermutation(block.permutation.withState("ns:lit", true))。onTick(e) などのイベントは e.block と e.dimension を持つ（currentTick は system.currentTick）。
- ワールド読み込み完了は world.afterEvents.worldLoad。
- スクリプトの最上位（読み込み直後）は early execution（初期化段階）: world.scoreboard / world.getDimension / world.getAllPlayers / setDynamicProperty などワールドの読み書きは不可（実行時エラーになる）。初期化処理は world.afterEvents.worldLoad.subscribe(() => { ... }) の中か system.run(() => { ... }) で行う。イベントの subscribe と system.runInterval の登録は最上位で可。
- 存在しない・削除済み: world.events / world.beforeEvents.worldInitialize / world.afterEvents.worldInitialize / chatSend（beta専用）/ runCommandAsync / world.executeCommand。
- beforeEvents のコールバックとカスタムコマンドのコールバックは読み取り専用の実行モード。ワールドを変更する処理（setDynamicProperty, setItem, teleport, spawnEntity, addEffect, フォームの show など）は system.run(() => { ... }) の中で行う。
- フォームの結果は await / then で受け取り、res.canceled なら何もしない。ActionFormResponse.selection / ModalFormResponse.formValues。
- インベントリ: player.getComponent("minecraft:inventory").container。手に持っているアイテムは container.getItem(player.selectedSlotIndex)。ItemStack を変更（setLore, nameTag, 耐久など）したら container.setItem(slot, item) で書き戻す。
- 動的プロパティの値は string | number | boolean | Vector3。オブジェクトは JSON.stringify / JSON.parse で保存する。読み出しは undefined の可能性を考える。
- スコアボード: world.scoreboard.getObjective(id) ?? world.scoreboard.addObjective(id, 表示名)。点数は objective.getScore(player) / setScore / addScore（Player には getScore は無い）。表示は world.scoreboard.setObjectiveAtDisplaySlot(DisplaySlotId.Sidebar, { objective })。
- ノックバック・吹き飛ばし（2.x）: entity.applyKnockback({ x: 方向x * 強さ, z: 方向z * 強さ }, 上向きの強さ)（引数は2つ。型定義の例にある4引数は1.x の古い形）。速度を直接足すなら entity.applyImpulse({ x, y, z })（プレイヤーには使えない）。
- エフェクト（2.x）: entity.addEffect("speed", 秒数 * 20, { amplifier: 0, showParticles: false })（時間は tick。強さは options の amplifier。4引数の古い形は不可）。エフェクトの ID は minecraft:speed / regeneration / resistance / jump_boost / night_vision / strength / slowness / instant_health など。
- 定期実行は system.runInterval(fn, tick) / 遅延は system.runTimeout(fn, tick)（20 tick = 1秒）。重い処理は system.runJob(function* () { ...; yield; }())。
- 時刻: world.getTimeOfDay()（0〜24000）、world.getDay()。アクションバー: player.onScreenDisplay.setActionBar(text)。
"""

# 日本語・英語のキーワード → 関係する宣言名
KEYWORDS: dict[str, list[str]] = {
    r"コマンド|/\w+": ["StartupEvent", "CustomCommandRegistry", "CustomCommand", "CustomCommandParameter",
                     "CustomCommandParamType", "CustomCommandOrigin", "CustomCommandResult", "CustomCommandStatus",
                     "CommandPermissionLevel"],
    r"メニュー|フォーム|UI|画面|ボタン|一覧から選": ["ActionFormData", "ActionFormResponse"],
    r"入力|テキスト|トグル|スライダー|ドロップダウン|設定画面|ModalForm": ["ModalFormData", "ModalFormResponse",
                                                                 "ModalFormDataTextFieldOptions"],
    r"確認|はい/いいえ|MessageForm": ["MessageFormData", "MessageFormResponse"],
    r"scriptevent|スクリプトイベント": ["ScriptEventCommandMessageAfterEvent"],
    r"保存|動的プロパティ|記録|データ": ["World", "Player"],
    r"lore|説明文|名前を付け|名前に色|アイテム": ["ItemStack", "Container", "EntityInventoryComponent"],
    r"倒し|倒す|死亡|討伐": ["EntityDieAfterEvent", "EntityDamageSource"],
    r"攻撃|殴|ダメージ": ["EntityHurtAfterEvent", "EntityHitEntityAfterEvent", "EntityDamageSource"],
    r"壊|採掘|掘": ["PlayerBreakBlockBeforeEvent", "PlayerBreakBlockAfterEvent", "Block", "BlockPermutation"],
    r"右クリック|クリック|触れ|インタラクト": ["PlayerInteractWithBlockBeforeEvent", "PlayerInteractWithBlockAfterEvent",
                                         "PlayerInteractWithEntityAfterEvent", "ItemUseAfterEvent"],
    r"使う|使った|使うと": ["ItemUseAfterEvent", "ItemComponentUseEvent", "ItemCustomComponent"],
    r"耐久": ["ItemDurabilityComponent", "ItemStack"],
    r"しゃが|スニーク": ["Player"],
    r"ノックバック|吹き飛": ["Entity", "VectorXZ"],
    r"周囲|範囲|近く": ["Dimension", "EntityQueryOptions"],
    r"アクションバー|タイトル|HUD|表示して": ["ScreenDisplay"],
    r"スコアボード|所持金|お金|送金|money": ["Scoreboard", "ScoreboardObjective", "DisplaySlotId",
                                       "ScoreboardObjectiveDisplayOptions"],
    r"参加|初めて|入った|スポーン": ["PlayerSpawnAfterEvent"],
    r"テレポート|移動させ": ["Entity", "TeleportOptions"],
    r"パーティクル|音|サウンド": ["Dimension", "Player"],
    r"燃や|炎|火": ["Entity"],
    r"エフェクト|回復|体力": ["Entity", "EntityHealthComponent"],
    r"インベントリ|手に持|持って|配っ|渡": ["EntityInventoryComponent", "Container", "EntityEquippableComponent",
                                     "EquipmentSlot", "ItemStack"],
    r"秒ごと|毎秒|定期|1日|時刻|時間": ["System"],
    r"ラグ|少しずつ|大量|runJob": ["System"],
    r"ブロックを(置|埋|設置)|埋め": ["Dimension", "BlockPermutation", "BlockVolume"],
    r"カスタムブロック|ブロック.*(コンポーネント|スクリプト)": ["BlockComponentRegistry", "BlockCustomComponent",
                                                    "BlockComponentPlayerInteractEvent"],
    r"カスタムコンポーネント|杖|アイテムのJSON": ["ItemComponentRegistry", "ItemCustomComponent", "ItemComponentUseEvent"],
    r"エンティティ.*右クリック|客|NPC": ["PlayerInteractWithEntityAfterEvent", "Entity"],
    r"クールダウン": ["System", "ItemCooldownComponent"],
}


@lru_cache(maxsize=1)
def member_owners() -> dict[str, list[str]]:
    """メンバー名 → それを持つ宣言名（"Property 'x' does not exist" の修正ヒント用）。"""
    owners = defaultdict(list)
    for name, d in load_index().items():
        for m in re.finditer(r"^\s+(?:readonly\s+|static\s+)?(\w+)\??[(:]", d.text, re.M):
            owners[m.group(1)].append(name)
    return owners


def retrieve(text: str, max_chars: int = 16000) -> list[str]:
    """依頼文（とコード）から関係する宣言名を選ぶ。"""
    names: list[str] = []
    for pat, decls in KEYWORDS.items():
        if re.search(pat, text, re.I):
            names += decls
    # 英語の API 名が直接書かれていればそれも
    idx = load_index()
    names += [w for w in re.findall(r"\b([A-Z]\w{3,})\b", text) if w in idx]
    # 言及されたイベント（world.afterEvents.xxx）の引数型
    for m in re.finditer(r"(?:afterEvents|beforeEvents)\.(\w+)", text):
        names += [n for n in idx if n.lower() in (m.group(1).lower() + "afterevent", m.group(1).lower() + "beforeevent")]
    seen, out, total = set(), [], 0
    for n in names:
        if n in seen or n not in idx:
            continue
        seen.add(n)
        size = min(idx[n].size, 6000)
        if total + size > max_chars:
            continue
        out.append(n)
        total += size
    return out


def build_context(text: str) -> str:
    decls = retrieve(text)
    sigs = "\n\n".join(s for s in (signature(n) for n in decls) if s)
    return (f"{CHEATSHEET}\n# イベント一覧（world / system）\n{event_table()}\n\n"
            f"# 関係しそうな API の型定義（@minecraft/server 2.10.0）\n```ts\n{sigs}\n```\n")


# よく出る型エラーと、その直し方（RAG の試行で qwen3-coder が繰り返し詰まったもの）
PATTERN_HINTS = [
    (r"'Entity' is not assignable to parameter of type 'Player'|does not exist on type 'Entity'",
     "Entity を Player として使っています。`if (!(entity instanceof Player)) return;` で Player か確認してから使ってください"
     "（Player を @minecraft/server から import）。カスタムコマンドの実行者 origin.sourceEntity や "
     "scriptEventReceive の ev.sourceEntity、damageSource.damagingEntity は Entity 型です。"),
    (r"'\"(\w+)\"' is not assignable to (parameter of )?type '(\w+)'",
     "文字列ではなく列挙型のメンバーを使ってください（例: CustomCommandParamType.String, "
     "CommandPermissionLevel.Admin, DisplaySlotId.Sidebar, EquipmentSlot.Mainhand）。"),
    (r"Argument of type '\d+' is not assignable to parameter of type '(\w+)'",
     "数値ではなく列挙型のメンバーを渡してください（例: DisplaySlotId.Sidebar, EquipmentSlot.Mainhand）。"),
    (r"types '\w+' and '\"\w+\"' have no overlap",
     "列挙型の値は文字列ではなくメンバーで比較してください。メンバー名は先頭が大文字です"
     "（例: player.getGameMode() === GameMode.Creative、weather === WeatherType.Rain、face === Direction.North）。"
     "その列挙型を @minecraft/server から import してください。"),
    (r"Type 'Boolean' has no call signatures|Type 'boolean' has no call signatures",
     "isSneaking・isSprinting・isOnGround・isFalling などはメソッドではなくプロパティです（player.isSneaking のように () を付けない）。"),
    (r"'(undefined|any|\w+)\[\]' is not assignable to parameter of type 'string \| number \| boolean \| Vector3'"
     r"|is not assignable to parameter of type 'string \| number \| boolean \| Vector3",
     "動的プロパティに保存できるのは string / number / boolean / Vector3 だけです。配列やオブジェクトは "
     "JSON.stringify(値) で文字列にして保存し、読むときに JSON.parse(String(x ?? \"[]\")) で戻してください。"),
    (r"Property 'brokenBlockPermutation' does not exist on type 'PlayerBreakBlockBeforeEvent'",
     "beforeEvents.playerBreakBlock では壊す前のブロックが ev.block（種類は ev.block.typeId）にあります。"
     "brokenBlockPermutation は afterEvents.playerBreakBlock にだけあります。"),
    (r"Property 'getSpawnLocation' does not exist on type 'Dimension'",
     "ワールドの初期スポーン地点は world.getDefaultSpawnLocation()、プレイヤーのリスポーン地点は player.getSpawnPoint() です。"),
    (r"'string \| number \| (boolean|true)( \| undefined)?' is not assignable to (parameter of )?type '(string|number)'",
     "フォームの formValues の各要素は string | number | boolean | undefined です。使う型に合わせて "
     "String(v) / Number(v) / Boolean(v) で変換してください（textField は文字列、slider と dropdown は数値、toggle は真偽値）。"),
    (r"Property '(setTimeOfDay|getTimeOfDay|setWeather)' does not exist on type 'World'|"
     r"Property '(setTimeOfDay|getTimeOfDay)' does not exist on type 'Dimension'",
     "時間は world.setTimeOfDay(TimeOfDay.Day や 0〜23999) / world.getTimeOfDay()、天気は dimension.setWeather(WeatherType.Clear) です。"),
    (r"Property '(createBlock|placeBlock|setBlock)' does not exist on type 'Dimension'",
     "ブロックを置くときは dimension.setBlockType({x,y,z}, \"minecraft:stone\")、またはブロックを取得して "
     "dimension.getBlock(pos)?.setType(\"minecraft:stone\") / setPermutation(BlockPermutation.resolve(\"minecraft:stone\")) を使います。"
     "範囲を埋めるなら dimension.fillBlocks(new BlockVolume(from, to), \"minecraft:stone\")。"),
    (r"does not exist on type 'typeof (BlockTypes|ItemTypes|EntityTypes)'",
     "ブロック・アイテム・エンティティの種類は文字列の ID（\"minecraft:gold_block\" など）で指定します。"
     "BlockTypes.get(\"minecraft:gold_block\") のように get で取得することもできます（BlockTypes.goldBlock のようなメンバーは無い）。"),
    (r"Expected 2-3 arguments, but got 4",
     "2.x の addEffect は addEffect(種類, 時間tick, { amplifier: 0, showParticles: true }) です"
     "（1.x の addEffect(種類, 時間, 強さ, 粒子) の形は使えない。時間は秒ではなく tick = 秒×20）。"),
    (r"Property 'setBlockType' does not exist on type 'Block'",
     "Block の種類を変えるのは block.setType(\"minecraft:gold_block\") または block.setPermutation(...) です"
     "（setBlockType は Dimension のメソッド）。"),
    (r"Property 'setComponent' does not exist on type 'Entity'",
     "アイテムを地面に落とすときは dimension.spawnItem(new ItemStack(\"minecraft:stick\", 1), location) を使います。"),
    (r"does not exist on type 'typeof ItemLockMode'",
     "ItemLockMode のメンバーは小文字です: ItemLockMode.slot（スロットに固定）/ ItemLockMode.inventory（インベントリ内に固定）/ ItemLockMode.none。"),
    (r"alias \[\w+\] already in use",
     "カスタムコマンド名が標準コマンド（give, tp, kill など）と重なっています。別の名前（例: ns:givekit）にしてください。"),
    (r"minecraft:(destroy_time|explosion_resistance|block_light_emission|block_light_absorption|unit_cube|creative_category|render_offsets)",
     "古い名前のコンポーネントです。ブロックは minecraft:destructible_by_mining { seconds_to_destroy }、"
     "minecraft:destructible_by_explosion { explosion_resistance }、minecraft:light_emission、minecraft:light_dampening、"
     "見た目は minecraft:geometry と minecraft:material_instances。アイテムの分類は description.menu_category { category } です。"),
    (r"description -> category|description/category",
     "アイテムの分類は description.menu_category: { \"category\": \"equipment\" } に書きます（description.category は無い）。"),
    (r"Property 'getBlockPermutation' does not exist on type 'Dimension'",
     "ブロックの状態は dimension.getBlock(pos)?.permutation で取得します。"),
    (r"'enumName' does not exist in type 'CustomCommandParameter'",
     "列挙型の引数は、startup で ev.customCommandRegistry.registerEnum(\"ns:mode\", [\"a\", \"b\"]) を登録し、"
     "パラメータは { type: CustomCommandParamType.Enum, name: \"ns:mode\" } と書きます（name に列挙型の名前）。"),
    (r"Property 'player' does not exist on type 'PlayerJoinAfterEvent'",
     "playerJoin には playerId / playerName しかありません（まだエンティティとして存在しない）。"
     "プレイヤーを使うなら world.afterEvents.playerSpawn（ev.player、初回だけなら ev.initialSpawn）を使ってください。"),
    (r"must be prefixed with a namespace",
     "カスタムコマンドや列挙型の名前には名前空間が必要です（例: \"my:skills\"）。"),
    (r"Property 'cancel' does not exist on type 'Entity\w*AfterEvent'",
     "afterEvents は起きた後の通知なのでキャンセルできません。ダメージを無効にしたいなら、受けた分を回復させる"
     "（health.setCurrentValue）か、エフェクト（resistance）で防いでください。"),
    (r"Property 'worldInitialize' does not exist",
     "worldInitialize は 1.x の古い API です。カスタムコンポーネントやコマンドの登録は "
     "system.beforeEvents.startup.subscribe((ev) => { ... })、ワールドの初期化は world.afterEvents.worldLoad です。"),
    (r"Property 'container' does not exist on type 'Block\w*'",
     "チェストなどの中身は block.getComponent(\"minecraft:inventory\")?.container で取得します。"),
    (r"Property 'entity' does not exist on type 'BlockRaycastHit'",
     "見ているエンティティは player.getEntitiesFromViewDirection({ maxDistance: 30 })[0]?.entity、"
     "見ているブロックは player.getBlockFromViewDirection({ maxDistance: 30 })?.block です。"),
    (r"Property 'getInteractionBlock' does not exist",
     "見ているブロックは player.getBlockFromViewDirection({ maxDistance: 8 })?.block で取得します。"),
    (r"not assignable to parameter of type 'BlockVolumeBase'",
     "fillBlocks には範囲のオブジェクトを渡します: dimension.fillBlocks(new BlockVolume(from, to), \"minecraft:stone\")"
     "（BlockVolume を @minecraft/server から import）。"),
    (r"minecraft:use_duration",
     "アイテムの使用時間は minecraft:use_modifiers { \"use_duration\": 秒数, \"movement_modifier\": 0.35 } に書きます。"),
    (r"Cannot find name '(Player|Entity|ItemStack|Block|BlockPermutation|EquipmentSlot|GameMode|Direction|"
     r"EntityDamageCause|CustomCommandParamType|CommandPermissionLevel|CustomCommandStatus|DisplaySlotId|BlockVolume)'",
     "使っているクラス・列挙型を import していません。ファイル先頭の import { ... } from \"@minecraft/server\" に追加してください。"),
    (r"minecraft:weapon has been deprecated",
     "minecraft:weapon は廃止されています。攻撃力は minecraft:damage、攻撃時の処理はスクリプトの "
     "world.afterEvents.entityHitEntity か、カスタムコンポーネントの onHitEntity で書きます。"),
    (r"minecraft:custom_components",
     "JSON の \"minecraft:custom_components\" は使いません（旧方式）。format_version \"1.21.90\" で、"
     "components の中にカスタムコンポーネント名をそのまま書きます: \"components\": { \"ns:name\": {}, ... }。"),
    (r"Property '(startup|system)' does not exist on type 'WorldBeforeEvents'",
     "startup は world ではなく system のイベントです: system.beforeEvents.startup.subscribe((ev) => { ... })。"),
    (r"'id' does not exist in type 'EntityQueryOptions'",
     "エンティティの種類で絞り込むときは { type: \"minecraft:zombie\" }、名前なら { name: \"…\" } です（id は無い）。"),
    (r"Property 'blockLocation' does not exist",
     "壊したブロックの位置は ev.block.location です（blockLocation は 1.x の古い名前）。"),
    (r"-> events: this member was found in the input",
     "アイテムの events は廃止されています。使ったときなどの処理はスクリプトのカスタムコンポーネント（onUse など）で書きます。"),
    (r"requires \"minecraft:use_modifiers > use_duration\"",
     "minecraft:food には \"minecraft:use_modifiers\": { \"use_duration\": 1.6 } も必要です。"),
    (r"Property 'getPlayer' does not exist on type 'World'",
     "名前でプレイヤーを探すときは world.getPlayers({ name: 名前 })[0] を使います（getPlayer は無い）。"),
    (r"Property '(playerChat|chatSend)' does not exist",
     "チャットのイベント（chatSend）は beta 専用で安定版にはありません。コマンドを作るなら "
     "system.beforeEvents.startup の ev.customCommandRegistry.registerCommand でカスタムコマンドにしてください。"),
    (r"Property 'explosion' does not exist on type 'Explosion(Before|After)Event'",
     "爆発イベントは ev.source（爆発させたエンティティ）、ev.dimension、ev.getImpactedBlocks()、"
     "beforeEvents なら ev.cancel / ev.setImpactedBlocks(blocks) を使います。"),
    (r"Property '\w+' does not exist on type 'Entity(Health|Attribute)Component'",
     "体力の最大値は health.effectiveMax（または defaultValue）、現在値は health.currentValue、設定は health.setCurrentValue(v) です。"),
    (r"does not exist on type '(Entity|Item|Block)Component'",
     "getComponent の戻り値が汎用の型になっています。ID を文字列リテラルで直接渡すと専用の型になります"
     "（例: getComponent(\"minecraft:health\") → EntityHealthComponent、getComponent(\"minecraft:inventory\") → "
     "EntityInventoryComponent、ItemStack は getComponent(\"minecraft:durability\")）。"
     "変数や存在しない ID を渡していないか、そのコンポーネントがそのプロパティを持つか型定義で確認してください"
     "（体力などの値は .value ではなく currentValue）。"),
    (r"ModalFormData\w*Options'",
     "2.x の ModalFormData は初期値などをオプションのオブジェクトで渡します: "
     "dropdown(label, items, { defaultValueIndex: 0 }), slider(label, min, max, { defaultValue: 5, valueStep: 1 }), "
     "toggle(label, { defaultValue: true }), textField(label, placeholder, { defaultValue: \"\" })。"),
    (r"Property '(itemComponentRegistry|blockComponentRegistry|customCommandRegistry)' does not exist",
     "これらのレジストリは system.beforeEvents.startup.subscribe((ev) => { ev.customCommandRegistry ... }) の ev にだけあります。"),
    (r"is not assignable to type 'CustomCommandResult'|CustomCommandResult",
     "カスタムコマンドのコールバックは { status: CustomCommandStatus.Success } か undefined を返します。"),
    (r"Object is possibly 'undefined'|possibly 'undefined'",
     "undefined の可能性がある値は `?.` や `if (!x) return;` で確認してから使ってください。"),
    (r"Operator '[+\-<>]=?' cannot be applied to types 'string \| number \| (boolean|true) \| Vector3'",
     "getDynamicProperty の戻り値は string | number | boolean | Vector3 | undefined です。"
     "数値として使うなら `Number(x ?? 0)` や `typeof x === \"number\" ? x : 0` で変換してください。"),
    (r"'string \| number \| (boolean|true) \| Vector3[^']*' is not assignable to (parameter of )?type '(string|number)'",
     "getDynamicProperty の値を文字列として使うときは `String(x ?? \"\")` または `typeof x === \"string\"` で確認してください。"
     "JSON を保存したなら `JSON.parse(String(x ?? \"{}\"))`。"),
    (r"cannot be used in early execution",
     "スクリプトの最上位（読み込み直後）でワールドを読み書きしています。その処理を "
     "world.afterEvents.worldLoad.subscribe(() => { ... }) の中に移してください（関数の中で使うなら、その関数が呼ばれるのが worldLoad 後か確認）。"),
    (r"does not have required privileges|cannot be used in restricted execution|in read-only mode",
     "beforeEvents（またはカスタムコマンド）のコールバックの中でワールドを変更しています。その処理を system.run(() => { ... }) で包んでください。"),
    (r"Cannot find module '\.\.?/",
     "存在しない自作モジュールを import しています。この依頼の答えは1つのファイルで完結させ、"
     "必要な関数や定数はそのファイルの中に書いてください（import は @minecraft/server と @minecraft/server-ui のみ）。"),
    (r"Expected \d+(-\d+)? arguments?, but got \d+",
     "引数の数が型定義と合っていません。上の型定義の宣言を見て、必須の引数を確認してください。"),
    (r"No overload matches this call",
     "どの呼び出し方にも合いません。引数の型（列挙型・オプションのオブジェクト・Vector3 {x,y,z} など）を型定義に合わせてください。"),
]


def repair_hints(errors: list[str]) -> str:
    """tsc のエラーから、正しい宣言やメンバーの所在を示すヒントを作る。"""
    hints, shown = [], set()
    owners = member_owners()
    joined = "\n".join(errors)
    for pat, hint in PATTERN_HINTS:
        if re.search(pat, joined):
            hints.append(f"- {hint}")
    for e in errors:
        m = re.search(r"Property '(\w+)' does not exist on type '(\w+)'", e)
        if m:
            prop, typ = m.groups()
            where = [o for o in owners.get(prop, []) if o != typ][:4]
            hints.append(f"- {typ} に {prop} はありません。" + (f"{prop} を持つのは: {', '.join(where)}" if where
                                                              else f"{typ} のメンバーから選んでください。"))
            if typ not in shown and signature(typ):
                shown.add(typ)
                hints.append(f"```ts\n{signature(typ, 4000)}\n```")
            continue
        m = re.search(r"has no exported member (?:named )?'(\w+)'", e)
        if m:
            name = m.group(1)
            d = load_index().get(name)
            hints.append(f"- {name} は {d.module} から import してください。" if d
                         else f"- {name} は存在しません。")
            continue
        m = re.search(r"Cannot find name '(\w+)'", e)
        if m:
            d = load_index().get(m.group(1))
            if d:
                hints.append(f"- {m.group(1)} を {d.module} から import してください。")
    return "\n".join(hints)
