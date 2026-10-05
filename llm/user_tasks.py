"""ユーザーの実際の開発スタイルに合わせた評価問題（「私がしそうな問題」）。

ユーザーの開発中パック（RandomEquipment / RestaurantEmpire / LargeStructure / ScriptEventSuggest /
MyCustomAddon / pve）で使われている機能・API・書き方の傾向から作成した。ユーザーのコードそのものは含まない。

採点（llm/eval_user.py）:
  - 出力からファイル（JSON / JS / TS）を取り出し、JSONは bedrock_tools、スクリプトは tsc で検証
  - require: 出力スクリプトに含まれるべき正規表現（このAPIを使って実装しているか）
  - forbid : 含まれてはいけない正規表現（2.xで削除された・beta専用のAPI など）
対象は @minecraft/server 2.10.0 / @minecraft/server-ui 2.2.0（安定版）。
"""
from __future__ import annotations

import json
from pathlib import Path

OLD_API = [r"worldInitialize", r"world\.events\.", r"\bchatSend\b", r"executeCommand", r"runCommandAsync"]

TASKS: list[dict] = [
    # ---------------- カスタムコマンド・UI（ScriptEventSuggest / MyCustomAddon 系）----------------
    {"id": "cmd_menu", "kind": "script", "from": "ScriptEventSuggest",
     "instruction": "/menu っていうカスタムコマンドを作って、実行したら ActionFormData でメインメニュー（ホーム・ショップ・設定）を開くようにして。誰でも使えるように。",
     "require": [r"system\.beforeEvents\.startup", r"customCommandRegistry\.registerCommand", r"ActionFormData", r"\.show\("],
     "forbid": OLD_API},
    {"id": "cmd_rank", "kind": "script", "from": "MyCustomAddon",
     "instruction": "ランク機能を作りたい。/rank add <プレイヤー> <ランク名> でOPだけがランクを付けられて、動的プロパティに保存。ネームタグは「[ランク] 名前」にして。",
     "require": [r"registerCommand", r"CommandPermissionLevel", r"setDynamicProperty", r"nameTag"],
     "forbid": OLD_API},
    {"id": "ui_home", "kind": "script", "from": "MyCustomAddon",
     "instruction": "ModalFormData で名前を入力して今いる場所をホームとして保存して、別のメニューで保存したホーム一覧から選んだらテレポートできるようにして。データはプレイヤーの動的プロパティにJSONで。",
     "require": [r"ModalFormData", r"textField", r"setDynamicProperty", r"teleport", r"JSON\.(parse|stringify)"],
     "forbid": OLD_API},
    {"id": "se_handler", "kind": "script", "from": "ScriptEventSuggest",
     "instruction": "/scriptevent rq:give <レアリティ> で、そのレアリティの剣を作って名前に色を付けて渡したい。レアリティは common, rare, epic, legendary。",
     "require": [r"scriptEventReceive", r"ItemStack", r"nameTag", r"\.id\b"],
     "forbid": OLD_API},
    {"id": "ui_settings", "kind": "script", "from": "ScriptEventSuggest",
     "instruction": "設定画面を ModalFormData で作って。トグルで「パーティクル表示」、スライダーで「音量 0〜10」、ドロップダウンで「難易度（かんたん/ふつう/むずかしい）」。保存はプレイヤーの動的プロパティ。キャンセルされたら何もしない。",
     "require": [r"ModalFormData", r"toggle", r"slider", r"dropdown", r"canceled", r"setDynamicProperty"],
     "forbid": OLD_API},

    # ---------------- RPG装備（RandomEquipment 系）----------------
    {"id": "lore_killcount", "kind": "script", "from": "RandomEquipment",
     "instruction": "剣でモブを倒したら、その剣の説明文（lore）に「討伐数: N」を書いてカウントを増やしていくようにして。",
     "require": [r"entityDie", r"getLore", r"setLore", r"setItem"],
     "forbid": OLD_API},
    {"id": "socket_fire", "kind": "script", "from": "RandomEquipment",
     "instruction": "loreに「炎上」って入ってる武器で攻撃したら、相手を3秒燃やして炎のパーティクルを出して。",
     "require": [r"entityHurt", r"getLore", r"setOnFire"],
     "forbid": OLD_API},
    {"id": "autosmelt", "kind": "script", "from": "RandomEquipment",
     "instruction": "loreに「自動精錬」があるツルハシで鉄鉱石か金鉱石を壊したら、普通のドロップの代わりにインゴットを落とすようにして。",
     "require": [r"playerBreakBlock", r"getLore", r"ItemStack", r"spawnItem"],
     "forbid": OLD_API},
    {"id": "durability_warn", "kind": "script", "from": "RandomEquipment",
     "instruction": "手に持ってる武器の耐久値が残り10%を切ったら、アクションバーで警告して音を鳴らして。同じアイテムで何度も鳴らないようにして。",
     "require": [r"minecraft:durability|ItemDurabilityComponent", r"setActionBar", r"playSound"],
     "forbid": OLD_API},
    {"id": "shield_wave", "kind": "script", "from": "RandomEquipment",
     "instruction": "盾を持ってしゃがんだら周囲5ブロックの敵をノックバックさせる技を作って。クールダウンは10秒で、残り時間はアクションバーに出して。",
     "require": [r"isSneaking", r"getEntities", r"applyKnockback", r"setActionBar"],
     "forbid": OLD_API},
    {"id": "random_stats", "kind": "script", "from": "RandomEquipment",
     "instruction": "装備を作るときにレアリティ（Common〜Mythic）を重み付き抽選して、レアリティごとに攻撃力ボーナスの範囲を変えたい。抽選と性能決定の関数をモジュールに分けて書いて。",
     "require": [r"export\s+(function|const)", r"Math\.random"],
     "forbid": OLD_API, "files": 2},

    # ---------------- 経営・経済（RestaurantEmpire 系）----------------
    {"id": "economy", "kind": "script", "from": "RestaurantEmpire",
     "instruction": "スコアボード money を作ってサイドバーに表示して。/pay <プレイヤー> <金額> で送金できるようにして、所持金が足りなかったらエラーメッセージ。",
     "require": [r"scoreboard", r"addObjective|getObjective", r"setObjectiveAtDisplaySlot", r"registerCommand"],
     "forbid": OLD_API},
    {"id": "hud", "kind": "script", "from": "RestaurantEmpire",
     "instruction": "1秒ごとに全プレイヤーのアクションバーに「所持金」と「ゲーム内の時刻（何時何分）」を表示して。所持金はスコアボード money から。",
     "require": [r"runInterval", r"setActionBar", r"getTimeOfDay|timeOfDay"],
     "forbid": OLD_API},
    {"id": "serve_customer", "kind": "script", "from": "RestaurantEmpire",
     "instruction": "客のエンティティ（myrest:customer）を右クリックしたとき、手に料理（myrest: で始まるアイテム）を持ってたら提供したことにして、料理を1個減らしてお金を+50して。",
     "require": [r"playerInteractWithEntity", r"minecraft:inventory|minecraft:equippable", r"myrest:customer"],
     "forbid": OLD_API},
    {"id": "daily_sales", "kind": "script", "from": "RestaurantEmpire",
     "instruction": "ゲーム内で1日たつごとに、その日の売上合計をチャットで発表してリセットして。売上はワールドの動的プロパティに貯めておく。",
     "require": [r"runInterval", r"getDynamicProperty", r"setDynamicProperty", r"getAbsoluteTime|getDay|getTimeOfDay"],
     "forbid": OLD_API},
    {"id": "stove_block", "kind": "mixed", "from": "RestaurantEmpire",
     "instruction": "カスタムブロック「コンロ」(myrest:stove) を作って。右クリックで調理メニューを開いて、生の牛肉を持ってたら10秒後にステーキができるようにして。ブロックのJSONとスクリプト両方ちょうだい。",
     "require": [r"registerCustomComponent|playerInteractWithBlock", r"ActionFormData", r"runTimeout"],
     "forbid": OLD_API, "json_kinds": ["block"]},

    # ---------------- 建築・範囲操作（LargeStructure 系）----------------
    {"id": "wand_select", "kind": "script", "from": "LargeStructure",
     "instruction": "木の斧でブロックを左クリックで1点目、右クリックで2点目を選択して座標を保存して。左クリックでブロックが壊れないようにして。",
     "require": [r"playerBreakBlock", r"cancel\s*=\s*true", r"playerInteractWithBlock", r"wooden_axe"],
     "forbid": OLD_API},
    {"id": "fill_job", "kind": "script", "from": "LargeStructure",
     "instruction": "選択した2点の範囲を指定したブロックで埋める /fillbig <ブロック> を作って。範囲が大きくてもラグらないように runJob で少しずつ置いて。",
     "require": [r"runJob", r"function\s*\*|yield", r"registerCommand"],
     "forbid": OLD_API},

    # ---------------- 便利機能（pve 系）----------------
    {"id": "storage_ui", "kind": "script", "from": "pve",
     "instruction": "/scriptevent storage:open で仮想倉庫を開いて、手に持ってるアイテムを預けたり、一覧から選んで取り出したりできるようにして。1ページ10個でページ送り付き。保存は動的プロパティ。",
     "require": [r"scriptEventReceive", r"ActionFormData", r"minecraft:inventory", r"setDynamicProperty"],
     "forbid": OLD_API},
    {"id": "first_join_kit", "kind": "script", "from": "pve",
     "instruction": "ワールドに初めて入ったプレイヤーにだけ、石の剣とパン16個を配って。2回目以降は配らないで。",
     "require": [r"playerSpawn", r"initialSpawn|getDynamicProperty|hasTag", r"ItemStack"],
     "forbid": OLD_API},
    {"id": "heal_wand", "kind": "mixed", "from": "MyCustomAddon",
     "instruction": "回復の杖アイテム（my:heal_wand）を作って。使うと体力が4回復してハートのパーティクルが出て、5秒クールダウン。アイテムのJSONとカスタムコンポーネントのスクリプトをちょうだい。",
     "require": [r"itemComponentRegistry\.registerCustomComponent", r"onUse", r"minecraft:health"],
     "forbid": OLD_API, "json_kinds": ["item"]},

    # ---------------- JSON（自作パックの entities / items / blocks 系）----------------
    {"id": "json_plate", "kind": "json", "from": "RestaurantEmpire",
     "instruction": "料理を置く「皿」ブロック（myrest:plate）を作って。当たり判定は薄く（高さ2ピクセルくらい）、ジオメトリは geometry.plate、テクスチャは plate。",
     "json_kinds": ["block"]},
    {"id": "json_food", "kind": "json", "from": "RestaurantEmpire",
     "instruction": "料理アイテム「カルボナーラ」（myrest:carbonara）を作って。満腹度8、満腹でも食べられる、スタックは16まで。",
     "json_kinds": ["item"]},
    {"id": "json_guardian", "kind": "json", "from": "treasure",
     "instruction": "宝箱を守るゴーレム（treasure:guardian）を作って。体力100、動きは遅め、プレイヤーを見つけたら殴りに来る、ノックバックしない、倒すとエメラルドを3〜5個落とす。",
     "json_kinds": ["entity"]},
    {"id": "json_toggle_block", "kind": "json", "from": "tesuitem",
     "instruction": "ON/OFFの状態を持つランプブロック（my:lamp）を作って。状態は my:lit（true/false）で、ONのときだけ明るさ15で光るようにパーミュテーションで切り替えて。",
     "json_kinds": ["block"]},
    {"id": "json_greatsword", "kind": "json", "from": "misanfa",
     "instruction": "「ルビーの大剣」（my:ruby_greatsword）を作って。攻撃力12、耐久値2000、エンチャントできる、手に持ったときに剣っぽく持つ。",
     "json_kinds": ["item"]},

    # ---------------- 移行・バグ修正（バージョン混在しているコード向け）----------------
    {"id": "migrate_initialize", "kind": "fix", "from": "RestaurantEmpire",
     "instruction": "昔のコードを @minecraft/server 2.x で動くように直して。",
     "input_code": "import { world, CommandPermissionLevel } from \"@minecraft/server\";\n\nworld.beforeEvents.worldInitialize.subscribe((ev) => {\n  ev.customCommandRegistry.registerCommand(\n    { name: \"myrest:open\", description: \"店のメニューを開く\", permissionLevel: CommandPermissionLevel.Any },\n    (origin) => {\n      origin.sourceEntity?.sendMessage(\"open\");\n      return undefined;\n    }\n  );\n});\n\nworld.afterEvents.worldInitialize.subscribe(() => {\n  world.sendMessage(\"loaded\");\n});\n",
     "require": [r"system\.beforeEvents\.startup", r"worldLoad"],
     "forbid": [r"worldInitialize"]},
    {"id": "fix_before_event", "kind": "fix", "from": "RandomEquipment",
     "instruction": "ブロックを壊したときにエラー（権限がない的なやつ）が出る。直して。",
     "input_code": "import { world } from \"@minecraft/server\";\n\nworld.beforeEvents.playerBreakBlock.subscribe((ev) => {\n  const count = (ev.player.getDynamicProperty(\"mined\") ?? 0) + 1;\n  ev.player.setDynamicProperty(\"mined\", count);\n  ev.player.sendMessage(`採掘数: ${count}`);\n});\n",
     "require": [r"system\.run|afterEvents\.playerBreakBlock"],
     "forbid": OLD_API},
    {"id": "fix_form_cancel", "kind": "fix", "from": "pve",
     "instruction": "メニューを閉じる（×ボタン）とエラーになる。直して。",
     "input_code": "import { world, system } from \"@minecraft/server\";\nimport { ActionFormData } from \"@minecraft/server-ui\";\n\nsystem.afterEvents.scriptEventReceive.subscribe((ev) => {\n  if (ev.id !== \"my:menu\") return;\n  const player = ev.sourceEntity;\n  new ActionFormData().title(\"メニュー\").button(\"A\").button(\"B\").show(player).then((res) => {\n    const labels = [\"A\", \"B\"];\n    player.sendMessage(`選択: ${labels[res.selection].toLowerCase()}`);\n  });\n});\n",
     "require": [r"canceled|selection\s*===\s*undefined|selection\s*==\s*null"],
     "forbid": OLD_API},
    {"id": "replace_chatsend", "kind": "fix", "from": "MyCustomAddon",
     "instruction": "beta の chatSend でチャットコマンド（!rank）を作ってたけど、安定版で動くようにカスタムコマンドに置き換えて。",
     "input_code": "import { world } from \"@minecraft/server\";\n\nworld.beforeEvents.chatSend.subscribe((ev) => {\n  if (ev.message === \"!rank\") {\n    ev.cancel = true;\n    const rank = ev.sender.getDynamicProperty(\"rank\") ?? \"なし\";\n    ev.sender.sendMessage(`あなたのランク: ${rank}`);\n  }\n});\n",
     "require": [r"customCommandRegistry\.registerCommand", r"system\.beforeEvents\.startup"],
     "forbid": [r"\bchatSend\b", r"worldInitialize"]},

    # ---------------- TypeScript（GitHubから拾ったTSプロジェクト向け）----------------
    {"id": "ts_cooldown", "kind": "script", "lang": "ts", "from": "RandomEquipment",
     "instruction": "TypeScriptで、プレイヤーごと・スキルごとのクールダウンを管理するクラス CooldownManager を書いて。残りtick数の取得と、使えるかどうかの判定ができるように。",
     "require": [r"class\s+CooldownManager", r"Map<", r":\s*(number|boolean)", r"currentTick"],
     "forbid": OLD_API},
]


for _t in TASKS:
    _t.setdefault("split", "dev")  # 上の31問は RAG のヒント調整などで出力を見てきたので「開発用」

# ---------------------------------------------------------------------------
# 本番テスト用（2026-10-02 追加）: 教材・ヒント調整・RAG の例には一切使わない
# newNSPVE / FactoryEmp の機能から作成
# ---------------------------------------------------------------------------
TEST_TASKS: list[dict] = [
    # newNSPVE: 戦闘・魔法・スタミナ
    {"id": "t_stamina", "kind": "script", "from": "newNSPVE",
     "instruction": "スタミナ（SP）システムを作って。最大100で、近接攻撃するたびに10減って、1秒ごとに5回復。SPが足りないときは攻撃したときに「SPが足りない」とアクションバーに出して。SPは動的プロパティで。",
     "require": [r"entityHitEntity|entityHurt", r"runInterval", r"getDynamicProperty", r"setDynamicProperty", r"setActionBar"],
     "forbid": OLD_API},
    {"id": "t_mana_potion", "kind": "script", "from": "newNSPVE",
     "instruction": "マナポーション（my:mana_potion）を飲み終わったら、プレイヤーのマナを30回復して（最大100）。マナは動的プロパティで、回復したらアクションバーに今のマナを表示。",
     "require": [r"itemCompleteUse", r"my:mana_potion", r"setDynamicProperty", r"setActionBar"],
     "forbid": OLD_API},
    {"id": "t_fire_wand", "kind": "script", "from": "newNSPVE",
     "instruction": "炎の杖（my:fire_wand）を右クリックしたら、見ている方向に小さい火の玉（minecraft:small_fireball）を飛ばして。マナを10使って、足りなかったら撃てないようにして。",
     "require": [r"itemUse", r"my:fire_wand", r"getViewDirection", r"spawnEntity", r"minecraft:projectile|shoot|applyImpulse"],
     "forbid": OLD_API},
    {"id": "t_charge_spell", "kind": "script", "from": "newNSPVE",
     "instruction": "杖（my:storm_staff）を2.5秒以上長押ししてから離したら、周囲8ブロックの敵に雷を落とす大技にして。短く離したときは何もしない。",
     "require": [r"itemStartUse", r"itemStopUse|itemReleaseUse", r"currentTick|Date\.now", r"getEntities", r"lightning_bolt"],
     "forbid": OLD_API},
    {"id": "t_lifesteal", "kind": "script", "from": "newNSPVE",
     "instruction": "loreに「吸血」がある武器で攻撃したら、与えたダメージの20%だけ攻撃した人の体力を回復して。最大体力は超えないように。",
     "require": [r"entityHurt", r"getLore", r"minecraft:health", r"damage"],
     "forbid": OLD_API},
    {"id": "t_accessory", "kind": "script", "from": "newNSPVE",
     "instruction": "インベントリのどこかに指輪（my:speed_ring）を持っていたら、移動速度上昇Iを付けるようにして。1秒ごとにチェックして、持っていなければ何もしない。",
     "require": [r"runInterval", r"minecraft:inventory", r"my:speed_ring", r"addEffect"],
     "forbid": OLD_API},
    {"id": "t_stop_time", "kind": "script", "from": "newNSPVE",
     "instruction": "時計を右クリックしたら、周囲10ブロックのモブ（プレイヤー以外）を5秒間その場に止めて。止めてる間は毎tick元の位置に戻して。",
     "require": [r"itemUse", r"minecraft:clock", r"getEntities", r"teleport", r"runInterval|runTimeout"],
     "forbid": OLD_API},
    {"id": "t_rare_drop", "kind": "script", "from": "newNSPVE",
     "instruction": "ゾンビを倒したら5%の確率でダイヤモンドを落とすようにして。プレイヤーが倒したときだけ。",
     "require": [r"entityDie", r"minecraft:zombie", r"Math\.random", r"spawnItem", r"ItemStack"],
     "forbid": OLD_API},
    {"id": "t_daily_bonus", "kind": "script", "from": "newNSPVE",
     "instruction": "ログインボーナスを作って。ワールドに入ったときに前回受け取ってから22時間たってたらエメラルド5個をあげて、まだならあと何時間かチャットで教えて。",
     "require": [r"playerSpawn", r"Date\.now", r"getDynamicProperty", r"setDynamicProperty", r"ItemStack"],
     "forbid": OLD_API},
    {"id": "t_daily_challenge", "kind": "script", "from": "newNSPVE",
     "instruction": "日替わりチャレンジで「ゾンビを10体倒す」を作って。達成したらお祝いのタイトルを出してスコアボード money に100足す。ゲーム内の日付が変わったらカウントをリセット。",
     "require": [r"entityDie", r"getDay|getAbsoluteTime", r"setTitle", r"scoreboard"],
     "forbid": OLD_API},
    {"id": "t_boss_scriptevent", "kind": "script", "from": "newNSPVE",
     "instruction": "/scriptevent boss:spawn <エンティティID> で、実行したプレイヤーの足元にそのボスを召喚して全員にチャットで告知して。IDが空なら my:frostlord。",
     "require": [r"scriptEventReceive", r"boss:spawn", r"spawnEntity", r"sendMessage"],
     "forbid": OLD_API},
    {"id": "t_summon_stone", "kind": "script", "from": "newNSPVE",
     "instruction": "召喚石（my:summon_stone）を右クリックしたら、目の前にボス（my:iron_golem_boss）を出して、召喚石を1個減らして。",
     "require": [r"itemUse", r"my:summon_stone", r"spawnEntity", r"setItem|amount"],
     "forbid": OLD_API},
    {"id": "t_menu_book", "kind": "script", "from": "newNSPVE",
     "instruction": "初めて参加したプレイヤーにメニューの本（my:menu_book）を渡して、それを右クリックしたらメニュー（ステータス・クエスト・ショップ）を開くようにして。",
     "require": [r"playerSpawn", r"initialSpawn|getDynamicProperty|hasTag", r"itemUse", r"ActionFormData"],
     "forbid": OLD_API},
    {"id": "t_quest_ui", "kind": "script", "from": "newNSPVE",
     "instruction": "クエスト一覧の画面を作って。クエストごとに進捗（例: 3/10）をボタンに表示して、押したら詳細を MessageFormData で見せて受注できるようにして。進捗は動的プロパティ。",
     "require": [r"ActionFormData", r"MessageFormData", r"getDynamicProperty", r"canceled"],
     "forbid": OLD_API},
    {"id": "t_shop_slider", "kind": "script", "from": "newNSPVE",
     "instruction": "ショップで買う個数をスライダーで選べるようにして（1〜64）、合計金額を確認画面で出して、OKなら money から引いてアイテムを渡して。お金が足りなければ買えない。",
     "require": [r"ModalFormData", r"slider", r"MessageFormData", r"scoreboard|getScore", r"ItemStack"],
     "forbid": OLD_API},
    {"id": "t_weapon_enchant", "kind": "script", "from": "newNSPVE",
     "instruction": "/scriptevent rq:enchant で、手に持ってる武器に成功率70%でランダムな独自エンチャント（会心・吸血・連撃のどれか）を付けてloreに書いて。失敗したらチャットで知らせて。",
     "require": [r"scriptEventReceive", r"Math\.random", r"setLore", r"setItem"],
     "forbid": OLD_API},

    # FactoryEmp: 経営・工場・採掘
    {"id": "t_company_create", "kind": "script", "from": "FactoryEmp",
     "instruction": "/fe:company <会社名> で会社を設立できるコマンドを作って。会社データ（名前・社長のID・資金0）はワールドの動的プロパティにJSONで保存して、もう会社があるならエラーにして。",
     "require": [r"registerCommand", r"setDynamicProperty", r"JSON\.stringify", r"CustomCommandParamType\.String"],
     "forbid": OLD_API},
    {"id": "t_contract", "kind": "script", "from": "FactoryEmp",
     "instruction": "契約掲示板の画面を作って。契約（例: 鉄インゴット32個で報酬500）を一覧から選んで、インベントリに数が揃ってたら回収して報酬を money に足して。足りなければ不足数を表示。",
     "require": [r"ActionFormData", r"minecraft:inventory", r"getItem|setItem", r"scoreboard|addScore"],
     "forbid": OLD_API},
    {"id": "t_market_sell", "kind": "script", "from": "FactoryEmp",
     "instruction": "売却画面を作って。インベントリにある鉱石（石炭・鉄・金・ダイヤ）を価格表に従って全部売って、合計金額を money に足して内訳をチャットに出して。",
     "require": [r"minecraft:inventory", r"setItem|clearAll", r"scoreboard|addScore", r"ActionFormData|MessageFormData"],
     "forbid": OLD_API},
    {"id": "t_machine_tick", "kind": "script", "from": "FactoryEmp",
     "instruction": "製錬機ブロック（fe:smelter）を置いたら座標を保存して、1秒ごとに全部の製錬機について、上に置いたチェストの鉄の原石を1個ずつ鉄インゴットに変えて。壊したら登録から消して。",
     "require": [r"playerPlaceBlock", r"playerBreakBlock", r"runInterval", r"getBlock", r"minecraft:inventory"],
     "forbid": OLD_API},
    {"id": "t_mine_reset", "kind": "script", "from": "FactoryEmp",
     "instruction": "/fe:mine_reset で、決まった範囲（32x30x32）の採掘場を作り直して。中身はランダムに石・石炭鉱石・鉄鉱石・ダイヤ鉱石。ラグらないように runJob で少しずつ置いて。",
     "require": [r"registerCommand", r"runJob", r"function\s*\*|yield", r"setType|setPermutation", r"Math\.random"],
     "forbid": OLD_API},
    {"id": "t_aoe_mining", "kind": "script", "from": "FactoryEmp",
     "instruction": "loreに「範囲採掘」があるツルハシでブロックを壊したら、周りの3x3（壊した面に沿って）もまとめて壊すようにして。岩盤は壊さない。",
     "require": [r"playerBreakBlock", r"getLore", r"getBlock|dimension", r"minecraft:bedrock"],
     "forbid": OLD_API},
    {"id": "t_spawn_guard", "kind": "script", "from": "FactoryEmp",
     "instruction": "採掘場の範囲（x 0〜32, z 0〜32）にモンスターが湧いたらすぐ消すようにして。",
     "require": [r"entitySpawn", r"remove\(\)"],
     "forbid": OLD_API},
    {"id": "t_ranking", "kind": "script", "from": "FactoryEmp",
     "instruction": "貢献ランキングの画面を作って。スコアボード contribution の点数が高い順に上位10人を、順位付きで表示して。",
     "require": [r"getObjective", r"getScores|getScore", r"sort", r"ActionFormData|MessageFormData"],
     "forbid": OLD_API},
    {"id": "t_research", "kind": "script", "from": "FactoryEmp",
     "instruction": "研究ツリーを作って。研究ごとに必要ポイントと前提研究があって、画面で選んで前提を満たしてればポイントを払って解放。解放済みはボタンに✔を付けて。データはワールドの動的プロパティ。",
     "require": [r"ActionFormData", r"getDynamicProperty", r"setDynamicProperty", r"JSON\.(parse|stringify)"],
     "forbid": OLD_API},
    {"id": "t_sidebar_hud", "kind": "script", "from": "FactoryEmp",
     "instruction": "サイドバーに会社の情報（所持金・電力・稼働中の機械の数）を表示して、5秒ごとに更新して。",
     "require": [r"setObjectiveAtDisplaySlot", r"addObjective|getObjective", r"setScore", r"runInterval"],
     "forbid": OLD_API},
    {"id": "t_rankup_fx", "kind": "script", "from": "FactoryEmp",
     "instruction": "会社のランクが上がったときの演出を作って。全員の画面に「ランクアップ！」のタイトルを出して、プレイヤーの周りに花火を3発打ち上げて音も鳴らして。関数 celebrateRankUp(rank) で呼べるように。",
     "require": [r"celebrateRankUp", r"setTitle", r"fireworks_rocket", r"playSound"],
     "forbid": OLD_API},

    # JSON
    {"id": "t_json_spear", "kind": "json", "from": "newNSPVE",
     "instruction": "武器の竜槍（my:dragon_spear）のアイテムJSONを作って。攻撃力9、耐久値1800、エンチャントできる、手に持ったときに武器っぽく持つ。",
     "json_kinds": ["item"]},
    {"id": "t_json_boss", "kind": "json", "from": "newNSPVE",
     "instruction": "ボス「フロストロード」（my:frostlord）のビヘイビアを作って。体力300、ノックバックしない、火に強い、プレイヤーを追いかけて殴る、自然には湧かない（召喚専用）、倒すとネザースターを落とす。",
     "json_kinds": ["entity"]},
    {"id": "t_json_machine", "kind": "json", "from": "FactoryEmp",
     "instruction": "工場の機械ブロック「圧縮機」（fe:press）を作って。壊すのに3秒、爆発耐性30、テクスチャは press、向きはプレイヤーの方を向くように置けるようにして。",
     "json_kinds": ["block"]},

    # 修正・移行
    {"id": "t_fix_runcommandasync", "kind": "fix", "from": "newNSPVE",
     "instruction": "古いアドオンのコードを今の @minecraft/server 2.x で動くように直して。",
     "input_code": "import { world } from \"@minecraft/server\";\n\nworld.afterEvents.itemUse.subscribe((ev) => {\n  if (ev.itemStack.typeId !== \"my:heal_orb\") return;\n  ev.source.runCommandAsync(\"effect @s regeneration 5 1 true\");\n  ev.source.runCommandAsync(\"playsound random.levelup @s\");\n});\n",
     "require": [r"\.runCommand\(|addEffect"],
     "forbid": [r"runCommandAsync", r"worldInitialize"]},
    {"id": "t_fix_dynprop_number", "kind": "fix", "from": "FactoryEmp",
     "instruction": "お金を足す関数で型エラーが出る。直して。",
     "input_code": "import { world } from \"@minecraft/server\";\n\n/**\n * @param {import(\"@minecraft/server\").Player} player\n * @param {number} amount\n */\nexport function addMoney(player, amount) {\n  const now = player.getDynamicProperty(\"money\") ?? 0;\n  const next = Math.max(0, now + amount);\n  player.setDynamicProperty(\"money\", next);\n  player.sendMessage(`所持金: ${next}`);\n}\n",
     "require": [r"Number\(|typeof\s+\w+\s*===\s*['\"]number['\"]|as number"],
     "forbid": OLD_API},
]
for _t in TEST_TASKS:
    _t["split"] = "test"
TASKS = TASKS + TEST_TASKS


def save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for t in TASKS:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    from collections import Counter

    out = Path(__file__).resolve().parent.parent / "data" / "eval" / "user_tasks.jsonl"
    save(out)
    print(out, len(TASKS), Counter(t["kind"] for t in TASKS), Counter(t["from"] for t in TASKS))
