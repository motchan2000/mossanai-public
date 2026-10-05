"""例示RAG: 依頼文に近い「お手本」（ユーザーの自己完結したコード）を探してプロンプトに添える。

お手本 = data/user/units.jsonl の standalone（単体で型チェックOK）な単位 + その逆算した依頼文。
検索は日本語に強い文字バイグラムの TF-IDF コサイン類似度（外部ライブラリ不要）。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from llm.fim import PROJECT_ROOT

UNITS = PROJECT_ROOT / "data" / "user" / "units.jsonl"
BACKTRANS = PROJECT_ROOT / "data" / "user" / "backtrans.jsonl"


@dataclass
class Example:
    id: str
    pack: str
    file: str
    request: str
    code: str


def _grams(text: str) -> Counter:
    t = re.sub(r"\s+", "", text.lower())
    return Counter(t[i:i + 2] for i in range(len(t) - 1))


@lru_cache(maxsize=1)
def load_examples() -> tuple[list[Example], list[dict[str, float]], dict[str, float]]:
    if not (UNITS.exists() and BACKTRANS.exists()):
        return [], [], {}
    requests = {json.loads(l)["id"]: json.loads(l)["request"]
                for l in BACKTRANS.read_text(encoding="utf-8").splitlines() if l}
    examples = []
    for line in UNITS.read_text(encoding="utf-8").splitlines():
        u = json.loads(line)
        uid = f"{u['pack']}:{u['file']}:{u['start']}"
        if u.get("standalone") and uid in requests:
            examples.append(Example(uid, u["pack"], u["file"], requests[uid], u["standalone"]))
    df = Counter()
    grams = []
    for e in examples:
        g = _grams(e.request)
        grams.append(g)
        df.update(set(g))
    n = len(examples)
    idf = {k: math.log((n + 1) / (v + 1)) + 1 for k, v in df.items()}
    vecs = [_tfidf(g, idf) for g in grams]
    return examples, vecs, idf


def _tfidf(g: Counter, idf: dict[str, float]) -> dict[str, float]:
    v = {k: c * idf.get(k, 1.0) for k, c in g.items()}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {k: x / norm for k, x in v.items()}


def api_terms(request: str) -> set[str]:
    """依頼から推定される API の識別子（イベント名・クラス名）。お手本のコードとの一致度に使う。"""
    from mcaddon.rag import retrieve

    terms = set()
    for name in retrieve(request):
        terms.add(name)
        m = re.match(r"(\w+?)(After|Before)Event$", name)  # EntityDieAfterEvent → entityDie
        if m:
            terms.add(m.group(1)[0].lower() + m.group(1)[1:])
    return terms


def find_examples(request: str, k: int = 2, exclude_packs: set[str] | None = None,
                  max_chars: int = 2500) -> list[Example]:
    """文字バイグラムの類似度（依頼文どうし）と、推定APIがお手本のコードに出てくる割合を半々で合算。"""
    examples, vecs, idf = load_examples()
    if not examples:
        return []
    q = _tfidf(_grams(request), idf)
    terms = api_terms(request)
    scored = []
    for e, v in zip(examples, vecs):
        if exclude_packs and e.pack in exclude_packs:
            continue
        if len(e.code) > max_chars:
            continue
        text_sim = sum(w * v.get(t, 0.0) for t, w in q.items())
        api_sim = (sum(1 for t in terms if re.search(rf"\b{t}\b", e.code)) / len(terms)) if terms else 0.0
        scored.append((0.5 * text_sim + 0.5 * api_sim, e))
    scored.sort(key=lambda x: -x[0])
    return [e for s, e in scored[:k] if s > 0.05]


def examples_block(request: str, k: int = 2, exclude_packs: set[str] | None = None) -> str:
    ex = find_examples(request, k, exclude_packs)
    if not ex:
        return ""
    parts = ["# 参考例（似た依頼と、型チェックを通る実装）"]
    for e in ex:
        parts.append(f"依頼: {e.request}\n```js\n{e.code.rstrip()}\n```")
    return "\n\n".join(parts) + "\n"
