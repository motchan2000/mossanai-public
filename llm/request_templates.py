"""ユーザーのコードからの逆算では足りない分野を補う、Script API の依頼文（テンプレート × 値の組み合わせ）。

    uv run python -m llm.request_templates   # → data/distill/template_requests.jsonl

イベント・UI・スコアボード・動的プロパティ・system・ワールド/ディメンション・エンティティ・
アイテム・ブロック（カスタムコンポーネント）・カスタムコマンドを、くだけた日本語の依頼として広く出す。
答えは蒸留（llm.distill）で先生モデルに書かせ、型チェック・実行テストを通ったものだけを使う。
"""
from __future__ import annotations

import json
import random

from llm.fim import PROJECT_ROOT

OUT = PROJECT_ROOT / "data" / "distill" / "template_requests.jsonl"

MOBS = [("ゾンビ", "minecraft:zombie"), ("スケルトン", "minecraft:skeleton"), ("クリーパー", "minecraft:creeper"),
        ("クモ", "minecraft:spider"), ("ウィッチ", "minecraft:witch"), ("ブレイズ", "minecraft:blaze")]
ITEMS = [("ダイヤモンド", "minecraft:diamond"), ("エメラルド", "minecraft:emerald"), ("鉄インゴット", "minecraft:iron_ingot"),
         ("金リンゴ", "minecraft:golden_apple"), ("パン", "minecraft:bread"), ("経験値ボトル", "minecraft:experience_bottle")]
EFFECTS = [("移動速度上昇", "speed"), ("再生", "regeneration"), ("耐性", "resistance"), ("跳躍力上昇", "jump_boost"),
           ("暗視", "night_vision"), ("力", "strength")]
BLOCKS = [("ダイヤモンドブロック", "minecraft:diamond_block"), ("金ブロック", "minecraft:gold_block"),
          ("ガラス", "minecraft:glass"), ("石", "minecraft:stone"), ("TNT", "minecraft:tnt")]
TOOLS = [("ダイヤの剣", "minecraft:diamond_sword"), ("鉄のツルハシ", "minecraft:iron_pickaxe"),
         ("弓", "minecraft:bow"), ("棒", "minecraft:stick"), ("ブレイズロッド", "minecraft:blaze_rod")]
NS = ["my", "rpg", "ex", "tw", "mod"]


def _p(rng: random.Random, xs):
    return rng.choice(xs)


def templates(rng: random.Random) -> list[tuple[str, str]]:
    """(分野, 依頼文) のリスト。毎回 rng から値を選ぶので、同じテンプレートでも中身が変わる。"""
    out: list[tuple[str, str]] = []

    def add(area: str, n: int, make):
        for _ in range(n):
            out.append((area, make()))

    # ---------------- afterEvents ----------------
    add("event", 4, lambda: (lambda m=_p(rng, MOBS), it=_p(rng, ITEMS):
        f"{m[0]}をプレイヤーが倒したら{rng.choice([5, 10, 20, 30])}%の確率で{it[0]}を落とすようにして。")())
    add("event", 3, lambda: (lambda it=_p(rng, ITEMS):
        f"プレイヤーが初めてワールドに入ったときだけ、{it[0]}を{rng.choice([1, 3, 5, 16])}個あげて歓迎メッセージを出して。")())
    add("event", 3, lambda: f"プレイヤーがワールドに参加・退出したら全員にチャットで「{rng.choice(['〇〇が参加しました', 'ようこそ〇〇さん'])}」みたいに知らせて。")
    add("event", 4, lambda: (lambda b=_p(rng, BLOCKS):
        f"プレイヤーが{b[0]}を壊したら、その場所に{rng.choice(['雷', '爆発', 'パーティクル', '経験値オーブ'])}を出して。")())
    add("event", 3, lambda: (lambda b=_p(rng, BLOCKS):
        f"プレイヤーが{b[0]}を置いたら、置いたプレイヤーにアクションバーで座標を表示して。")())
    add("event", 3, lambda: f"プレイヤーがダメージを受けたら、残り体力をアクションバーに「HP: 残り/最大」で出して。")
    add("event", 3, lambda: (lambda m=_p(rng, MOBS):
        f"プレイヤーが{m[0]}を殴ったら、{rng.choice([2, 3, 5])}秒間燃やして。")())
    add("event", 3, lambda: (lambda t=_p(rng, TOOLS):
        f"{t[0]}を持って右クリック（使用）したら、見ている方向に{rng.choice([5, 8, 10])}ブロックテレポートして。")())
    add("event", 3, lambda: (lambda it=_p(rng, ITEMS[3:5]):
        f"{it[0]}を食べ終わったら、{_p(rng, EFFECTS)[0]}を{rng.choice([10, 30, 60])}秒つけて。")())
    add("event", 3, lambda: "矢がエンティティに当たったら、当たった相手を少し上に打ち上げて。")
    add("event", 2, lambda: "投げた雪玉がブロックに当たったら、そのブロックの上に小さな爆発を起こして（地形は壊さない）。")
    add("event", 2, lambda: "ボタンが押されたら、押した人に「ボタンが押されました」とメッセージを出して、近くの鉄のドアの位置を表示して。")
    add("event", 2, lambda: "レバーをONにしたら時間を昼に、OFFにしたら夜にして。")
    add("event", 2, lambda: "感圧板を踏んだプレイヤーにジャンプ力上昇を5秒付けて、上に飛ばして。")
    add("event", 2, lambda: "天気が雨に変わったらチャットで全員に知らせて。")
    add("event", 2, lambda: "爆発が起きたら、爆発に巻き込まれたブロックの数をチャットで表示して。")
    add("event", 2, lambda: "プレイヤーがネザーやエンドに移動したら、移動先のディメンション名をタイトルで表示して。")
    add("event", 2, lambda: "プレイヤーがリスポーンしたら、3秒間の耐性を付けて「復活しました」と表示して。")
    add("event", 2, lambda: (lambda m=_p(rng, MOBS):
        f"{m[0]}がスポーンしたら、名前を「強化{m[0]}」にして体力を少しだけ回復させて。")())
    add("event", 2, lambda: "プレイヤーがエンティティを右クリックしたら、その相手のIDと体力をチャットに表示して。")
    add("event", 2, lambda: "プレイヤーがチェストを右クリックしたら、その座標を記録して「チェストを開いた回数」を数えて。")
    add("event", 2, lambda: "誰かに効果（エフェクト）が付いたら、付いたエフェクトの名前と時間をそのプレイヤーに表示して。")

    # ---------------- beforeEvents（キャンセル） ----------------
    add("before", 3, lambda: (lambda b=_p(rng, BLOCKS):
        f"{b[0]}はクリエイティブ以外では壊せないようにして、壊そうとしたら「このブロックは壊せません」と出して。")())
    add("before", 3, lambda: (lambda t=_p(rng, TOOLS):
        f"タグ「{rng.choice(['vip', 'admin', 'mage'])}」を持っていないプレイヤーは{t[0]}を使えないようにして。")())
    add("before", 2, lambda: "スポーン地点の周り半径30ブロックでは爆発でブロックが壊れないようにして。")
    add("before", 2, lambda: "看板やチェストを右クリックしようとしたとき、スニークしていなかったらキャンセルして警告して。")
    add("before", 2, lambda: "プレイヤーが退出するときに、その人の所持金（スコアボード money）をワールドの動的プロパティに保存して。")
    add("before", 2, lambda: "特定のエリア（x 0〜50, z 0〜50）ではブロックを置けないようにして。OPは除く。")

    # ---------------- UI ----------------
    add("ui", 4, lambda: f"/scriptevent {_p(rng, NS)}:menu で ActionFormData のメニューを開いて、ボタンは「{rng.choice(['回復', 'テレポート', 'ショップ'])}」「ステータス」「閉じる」にして、それぞれの処理も作って。")
    add("ui", 3, lambda: "時計を右クリックしたら ActionFormData のメニューを開いて、ボタンにアイコン（textures/items/...）も付けて。")
    add("ui", 3, lambda: f"ModalFormData でプレイヤー名のドロップダウンと金額の入力欄を出して、選んだ人に money を送金できるようにして。")
    add("ui", 3, lambda: "MessageFormData で「本当にリセットしますか？」と確認して、はいならスコアをリセットして。")
    add("ui", 3, lambda: f"ModalFormData の設定画面で、トグル・スライダー（{rng.choice(['1〜10', '0〜100', '5〜60'])}）・テキスト入力の値をプレイヤーの動的プロパティに保存して、次に開いたときは前回の値を初期値にして。")
    add("ui", 2, lambda: "フォームを開いたときにチャット画面などでふさがっていたら、開けるまで何回か再試行して。")
    add("ui", 2, lambda: "ショップ画面を作って。商品リスト（名前・ID・値段）から ActionFormData のボタンを作り、選んだら money を払ってアイテムを渡して。")
    add("ui", 2, lambda: "テレポートメニューを作って。オンラインのプレイヤー一覧をボタンにして、選んだ人のところへテレポート。")

    # ---------------- スコアボード ----------------
    add("score", 3, lambda: f"スコアボード「{rng.choice(['kills', 'points', 'coins'])}」がなければ作って、サイドバーに表示して。モブを倒すたびに1増やして。")
    add("score", 3, lambda: "スコアボード money の上位5人を、1分ごとにチャットでランキング表示して。")
    add("score", 2, lambda: "プレイヤーが死んだらスコアボード deaths を1増やして、死亡回数をタイトルで見せて。")
    add("score", 2, lambda: "毎日最初にログインしたら money に100足すログインボーナスを作って。日付はワールドの日数で判定して。")
    add("score", 2, lambda: "プレイ時間を秒単位でスコアボード playtime に記録して、リストに表示して。")

    # ---------------- 動的プロパティ ----------------
    add("dynprop", 3, lambda: f"プレイヤーごとにレベルと経験値を動的プロパティで持って、モブを倒すと経験値{rng.choice([5, 10, 15])}、100たまるとレベルアップしてお祝いの音を鳴らして。")
    add("dynprop", 3, lambda: "ワールドの動的プロパティに JSON で「ワープ地点の一覧」を保存・読み込みするユーティリティ関数を作って。")
    add("dynprop", 2, lambda: "アイテムに動的プロパティで「強化レベル」を持たせて、/scriptevent で手に持った武器を強化して名前に +N を付けて。")
    add("dynprop", 2, lambda: "エンティティに動的プロパティで飼い主の名前を保存して、右クリックで飼い主を表示して。")

    # ---------------- system ----------------
    add("system", 3, lambda: f"{rng.choice([1, 5, 10])}秒ごとに全プレイヤーの足元が{_p(rng, BLOCKS)[0]}なら{_p(rng, EFFECTS)[0]}を付けて。")
    add("system", 3, lambda: f"/scriptevent {_p(rng, NS)}:countdown <秒数> でカウントダウンをタイトルに出して、0になったら花火の音を鳴らして。")
    add("system", 2, lambda: "大量のブロックを置く処理を system.runJob とジェネレーター関数で少しずつ実行して、ラグを出さないようにして。")
    add("system", 2, lambda: "runInterval で動いているループを、/scriptevent で止めたり再開したりできるようにして。")
    add("system", 2, lambda: "ワールドの起動直後（早期実行）に触れない API があるので、ワールド読み込み後に初期化処理をする形で書いて。")

    # ---------------- ワールド・ディメンション ----------------
    add("world", 3, lambda: (lambda m=_p(rng, MOBS):
        f"/scriptevent {_p(rng, NS)}:wave でプレイヤーの周りに{m[0]}を{rng.choice([3, 5, 8])}体スポーンさせて。")())
    add("world", 3, lambda: "プレイヤーが見ているブロック（レイキャスト）の種類と座標をアクションバーにずっと表示して。")
    add("world", 2, lambda: "プレイヤーが見ている先のエンティティに雷を落とす杖を作って（最大30ブロック）。")
    add("world", 3, lambda: f"半径{rng.choice([5, 10, 16])}ブロック以内にいるモンスターの数を数えてアクションバーに表示して。")
    add("world", 2, lambda: "/scriptevent で時間を朝にして、天気を晴れにするコマンドを作って。")
    add("world", 2, lambda: (lambda b=_p(rng, BLOCKS):
        f"プレイヤーの足元に{b[0]}で5x5の床を作る関数を作って。")())
    add("world", 2, lambda: "指定した座標に構造物（structure）を設置する /scriptevent を作って。")
    add("world", 2, lambda: "プレイヤーの周りに円形にパーティクルを出すエフェクトを毎tick表示して。")

    # ---------------- エンティティ・プレイヤー ----------------
    add("entity", 3, lambda: (lambda e=_p(rng, EFFECTS):
        f"スニークしながらジャンプしたら{e[0]}を{rng.choice([3, 5, 10])}秒付ける技を作って。クールダウンは10秒。")())
    add("entity", 3, lambda: "プレイヤーの体力が最大体力の30%以下になったら、画面に赤いタイトルで警告して再生を付けて。")
    add("entity", 2, lambda: "近くの敵をまとめてノックバックさせる衝撃波スキルを作って。")
    add("entity", 2, lambda: "プレイヤーのインベントリの中身を全部チャットに一覧表示して。")
    add("entity", 2, lambda: "プレイヤーの防具（頭・胴・脚・足）を調べて、全部ダイヤ装備ならタグ full_diamond を付けて。")
    add("entity", 2, lambda: "タグ frozen が付いているプレイヤーは動けないようにして（毎tick元の位置に戻す）。")
    add("entity", 2, lambda: "ペット用に、プレイヤーの近くにいないオオカミを3秒ごとにプレイヤーのそばへテレポートさせて。")
    add("entity", 2, lambda: "落下ダメージを受けたプレイヤーに、受けたダメージ量をアクションバーで表示して。")
    add("entity", 2, lambda: "プレイヤーを前方に勢いよく飛ばすダッシュ技を作って（applyKnockback か applyImpulse を使って）。")

    # ---------------- アイテム ----------------
    add("item", 3, lambda: (lambda t=_p(rng, TOOLS):
        f"/scriptevent {_p(rng, NS)}:give で、名前と説明文（lore）付きの{t[0]}を渡して。説明文は2行。")())
    add("item", 3, lambda: "手に持った道具の耐久値を最大まで回復する /scriptevent を作って。")
    add("item", 2, lambda: "手に持った武器にエンチャント（鋭さ III）を付ける /scriptevent を作って。付けられない場合はメッセージを出して。")
    add("item", 2, lambda: "インベントリにある特定のアイテム（ダイヤ）の数を数えて、10個以上なら10個消費して報酬をあげて。")
    add("item", 2, lambda: "死んでもなくならない（keepOnDeath）アイテムを渡すコマンドを作って。")
    add("item", 2, lambda: "アイテムをスロットに固定（lockMode）して動かせないようにしたメニュー用アイテムを渡して。")
    add("item", 3, lambda: f"カスタムアイテム {_p(rng, NS)}:{rng.choice(['magic_wand', 'heal_staff', 'teleport_orb'])} のカスタムコンポーネントを Script で登録して、使ったら効果を出して。アイテムの JSON も書いて。")
    add("item", 2, lambda: "アイテムのクールダウンコンポーネントを使って、使った後に見た目のクールダウンを出して。")

    # ---------------- ブロック ----------------
    add("block", 3, lambda: f"カスタムブロック {_p(rng, NS)}:{rng.choice(['trap', 'jump_pad', 'heal_pad'])} を作って、上に乗ったら効果が出るようにカスタムコンポーネントで書いて。ブロックの JSON も。")
    add("block", 3, lambda: f"カスタムブロック {_p(rng, NS)}:{rng.choice(['switch', 'counter', 'lamp'])} を右クリックするたびに状態（ブロックステート）が切り替わるようにして。JSON も書いて。")
    add("block", 2, lambda: "カスタムブロックが一定間隔（onTick）で周りのプレイヤーを回復するようにして。JSON も書いて。")
    add("block", 2, lambda: "クリックしたブロックの向き（cardinal_direction などのステート）を調べてチャットに出して。")
    add("block", 2, lambda: "プレイヤーが右クリックしたチェストの中身を数えて表示して。")
    add("block", 2, lambda: "BlockPermutation を使って、指定した座標を上向きのハーフブロックにする関数を作って。")

    # ---------------- カスタムコマンド ----------------
    add("command", 3, lambda: f"/{_p(rng, NS)}:heal カスタムコマンドを作って、実行した人の体力を全回復して。OPだけ使えるように。")
    add("command", 3, lambda: f"/{_p(rng, NS)}:spawnmob <エンティティID> <数> のカスタムコマンドを作って、実行者の前にスポーンさせて。")
    add("command", 2, lambda: "/tpa <プレイヤー> のカスタムコマンドを作って、相手にフォームで許可をとってからテレポートさせて。")
    add("command", 2, lambda: "列挙型（enum）の引数を持つカスタムコマンド /mode <survival|creative> を作って、ゲームモードを変えて。")
    add("command", 2, lambda: "カスタムコマンドのコールバック内でワールドを変更するときは system.run で後回しにする形で /setspawn を作って。")

    # ---------------- 移行・TS ----------------
    add("ts", 3, lambda: f"TypeScript で、{rng.choice(['プレイヤーのステータス', 'パーティー', 'ショップの在庫'])}を管理するクラスを作って、データは動的プロパティに保存して。")
    add("ts", 2, lambda: "TypeScript で Vector3 の足し算・引き算・距離を計算するユーティリティを作って。")

    # ---------------- ゲームシステム（RPG・経済・工場・魔法・PvE） ----------------
    G = [
        ("rpg", "職業システムを作って。/scriptevent で戦士・魔法使い・弓使いを選べて、職業ごとに常時エフェクトを付けて。職業はプレイヤーの動的プロパティに保存。"),
        ("rpg", "モブを倒すと経験値の代わりに「ソウル」がたまって、メニューからソウルを使って攻撃力（力のエフェクト）を買えるようにして。"),
        ("rpg", "武器のレア度（コモン〜レジェンド）を名前の色で表して、レジェンド武器で攻撃したら追加で雷が落ちるようにして。"),
        ("rpg", "パーティー機能を作って。/scriptevent で招待・参加・脱退ができて、パーティーメンバー同士は攻撃してもダメージを無効化して。"),
        ("rpg", "クエストを作って。「スケルトンを5体倒す」を受けて、達成したらエメラルド10個の報酬。進捗はアクションバーに出して。"),
        ("rpg", "スキルポイントを振り分ける画面を ModalFormData のスライダーで作って、体力・攻撃・速度に割り振れるようにして。"),
        ("rpg", "ボスの体力をボスバーの代わりにアクションバーで「■■■□□」みたいなゲージで近くのプレイヤーに表示して。"),
        ("rpg", "クリティカルヒットを作って。攻撃したとき15%の確率でダメージを追加で与えて、クリティカルの文字とパーティクルを出して。"),
        ("rpg", "防具セットボーナスを作って。鉄装備を4つ全部着ていたら耐性を付けて、1つでも外したら消して。"),
        ("rpg", "敵を倒すとたまにアイテム「ポーションのかけら」を落として、5個集めると回復ポーションと交換できるようにして。"),
        ("econ", "銀行を作って。ATM ブロックを右クリックするとフォームが開いて、money を預ける・引き出すができて、預金はワールドの動的プロパティに保存。"),
        ("econ", "プレイヤー同士のトレード画面を作って。相手を選んで、渡すアイテムと値段を決めて、相手が承認したら交換。"),
        ("econ", "1時間（ゲーム内1日）ごとに、全員の預金に1%の利息を付けて。"),
        ("econ", "オークション機能を作って。手に持ったアイテムを出品して、他の人が入札できて、60秒後に一番高い人が落札。"),
        ("econ", "給料システムを作って。タグ worker を持っているプレイヤーには5分ごとに money を50あげて。"),
        ("econ", "売却箱を作って。特定のチェストに入れたアイテムを1分ごとに査定して money に換えて、チェストを空にして。"),
        ("factory", "採掘機ブロックを作って。置いた座標を保存して、10秒ごとに真下のブロックを1つ掘って上のチェストに入れて。"),
        ("factory", "発電機と電力のしくみを作って。発電機ブロックごとに毎秒10の電力をためて、ワールドの動的プロパティに記録して。"),
        ("factory", "ベルトコンベアを作って。特定のブロックの上に乗ったアイテム（item エンティティ）を、ブロックの向きの方向へ少しずつ動かして。"),
        ("factory", "自動かまどを作って。チェストに入った生肉を一定時間ごとに焼き肉に変えて。"),
        ("factory", "工場の生産量を1分ごとに集計して、オーナーのプレイヤーにチャットで報告して。"),
        ("magic", "マナシステムを作って。最大100、毎秒2回復、アクションバーにマナを表示して。魔法を使うとマナを消費。"),
        ("magic", "氷の杖を作って。右クリックで見ている方向にレイキャストして、当たった敵に鈍化を付けて、周りに雪のパーティクルを出して。"),
        ("magic", "回復魔法を作って。スニーク＋右クリックで半径5ブロックの味方プレイヤー全員の体力を4回復。"),
        ("magic", "テレポートの巻物を作って。使うと保存しておいたホームに5秒後にテレポート。途中で動いたらキャンセル。"),
        ("magic", "メテオ魔法を作って。見ている地点の上空から火の玉を3つ落として。"),
        ("magic", "魔法陣を作って。プレイヤーの足元に円と星形のパーティクルを数秒間描いて。"),
        ("magic", "詠唱システムを作って。アイテムを使い始めてから離すまでの時間で魔法の強さを変えて。"),
        ("pve", "ウェーブ制の防衛戦を作って。/scriptevent で開始すると、30秒ごとに敵が増えながらスポーンして、10ウェーブで終了。"),
        ("pve", "闘技場を作って。指定エリアに入ったプレイヤーの数を数えて、2人以上でカウントダウン後に戦闘開始。"),
        ("pve", "ボスが体力半分を切ったら第2形態として、周りに手下のゾンビを4体呼んで、移動速度を上げて。"),
        ("pve", "敵モブが倒されたときに、最後に攻撃したプレイヤーとダメージの合計をチャットに出して。"),
        ("pve", "リスキル防止で、リスポーンしたプレイヤーを5秒間無敵（耐性255）にして、その間は攻撃もできないようにして。"),
        ("pve", "ダンジョンの宝箱を作って。初めて開けたプレイヤーにだけランダムな報酬をあげて、開けた人は記録しておく。"),
        ("pve", "敵のスポナーを作って。カスタムブロックの周りにプレイヤーがいる間だけ、5秒ごとにモブを出して。最大10体まで。"),
        ("util", "プレイヤーの座標を「x, y, z」の整数でフォーマットする関数と、2点間の距離を求める関数を作って。"),
        ("util", "ランダムな整数・配列からランダムに1つ選ぶ・重み付き抽選をする関数をまとめたユーティリティを作って。"),
        ("util", "秒数を「1時間2分3秒」みたいな文字列に変換する関数を作って、プレイ時間表示に使って。"),
        ("util", "プレイヤー名からオンラインのプレイヤーを探す関数を作って。見つからなければ undefined。"),
        ("util", "デバッグ用に、/scriptevent debug:info で手に持ったアイテムのID・数・lore・コンポーネント一覧を表示して。"),
        ("util", "ワールドの動的プロパティのキー一覧と使用バイト数を表示する管理用コマンドを作って。"),
        ("util", "エラーが起きてもスクリプト全体が止まらないように、イベント処理を try/catch で包むヘルパー関数を作って。"),
    ]
    for area, req in G:
        out.append((area, req))
    return out


def main() -> None:
    rng = random.Random(42)
    rows, seen = [], set()
    for area, req in templates(rng) + templates(random.Random(7)):
        if req in seen:
            continue
        seen.add(req)
        rows.append({"id": f"tpl:{len(rows):04d}", "area": area, "request": req})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"{len(rows)} 件 → {OUT}")


if __name__ == "__main__":
    main()
