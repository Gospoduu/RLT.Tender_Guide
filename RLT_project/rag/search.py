from __future__ import annotations
import json, math, re
from dataclasses import dataclass
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

CHUNKS_JSONL = Path(__file__).parent / "data" / "chunks.jsonl"

# ---------- токены / стоп-слова ----------
TOKEN = re.compile(r"[a-zа-яё0-9]+", re.I)
STOP = {
    "и","в","во","на","по","к","ко","от","до","с","со","как","при","для","о","об","обо",
    "что","это","не","да","или","ли","из","над","под","без","уж","бы","же","то","а","но",
}

ALIASES = {
    "еис": ["единая", "информационная", "система"],
    "еруз": ["единый", "реестр", "участников", "закупок"],
    "змо": ["закупк", "мал", "объем", "объём"],
    "нмцк": ["начальн", "максимальн", "цена", "контракт"],
}

LAW_RX  = re.compile(r"\b(44\s*-\s*фз|223\s*-\s*фз|615\s*пп\s*рф)\b", re.I)
ROLE_RX = re.compile(r"\b(поставщик|заказчик|комиссия|оператор|покупатель|продавец)\b", re.I)
TOP_RX  = re.compile(r"\b(регистрац|аккредитац|змо|запрос предложен|жалоб|нмцк|срок|требован|штраф|договор|контракт)\w*", re.I)

def tok(text: str) -> List[str]:
    return [t.lower() for t in TOKEN.findall(text)]

def norm_query(q: str) -> List[str]:
    base = [t for t in tok(q) if t not in STOP]
    # расширяем аббревиатуры
    ext = []
    for t in base:
        ext.extend(ALIASES.get(t, []))
    return base + ext

def parse_context(q: str):
    law = None
    m = LAW_RX.search(q)
    if m:
        law = m.group(1).upper().replace(" ", "")
    role = (ROLE_RX.search(q) or [None])[0]
    topic = (TOP_RX.search(q) or [None])[0]
    role  = role.lower() if role else None
    topic = topic.lower() if topic else None
    return {"law": law, "role": role, "topic": topic}

# ---------- данные ----------
@dataclass
class Doc:
    id: int
    title: str
    url: str
    heading: str
    section_path: str
    text: str
    law_hint: Optional[str]
    role: Optional[str]
    topic: Optional[str]

DOCS: List[Doc] = []

# инвертированный индекс: term -> [(doc_id, tf, [pos...])]
POST: Dict[str, List[Tuple[int,int,List[int]]]] = defaultdict(list)
DF: Dict[str,int] = defaultdict(int)
DL: List[int] = []  # длина документа в токенах
AVGDL = 0.0
BUILT = False

def build_index(force=False):
    # === Читает chunks.jsonl и строит индекс BM25 + позиции для proximity.===
    global BUILT, AVGDL, DOCS, POST, DF, DL
    if BUILT and not force:
        return
    DOCS.clear(); POST.clear(); DF.clear(); DL.clear()

    if not CHUNKS_JSONL.exists():
        raise RuntimeError(f"Нет файла с чанками: {CHUNKS_JSONL}")

    with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            DOCS.append(Doc(
                id=i,
                title=row.get("title",""),
                url=row.get("url",""),
                heading=row.get("heading","Без подзаголовка"),
                section_path=row.get("section_path", row.get("title","")),
                text=row.get("text",""),
                law_hint=row.get("law_hint"),
                role=row.get("role"),
                topic=row.get("topic"),
            ))

    # индекс
    for d in DOCS:
        terms = tok(d.text)
        DL.append(len(terms))
        pos_map: Dict[str, List[int]] = defaultdict(list)
        for p, t in enumerate(terms):
            pos_map[t].append(p)
        cnt = {t: len(ps) for t, ps in pos_map.items()}
        for t, tf in cnt.items():
            POST[t].append((d.id, tf, pos_map[t]))
        for t in cnt.keys():
            DF[t] += 1

    AVGDL = (sum(DL) / len(DL)) if DL else 0.0
    BUILT = True
    print(f"[RAG] index: docs={len(DOCS)}, terms={len(DF)}, avgdl={AVGDL:.1f}")

# ---------- BM25 + контекстные бусты ----------
k1, b = 1.5, 0.75

def bm25_score(q_terms: List[str], doc_id: int) -> float:
    score = 0.0
    dl = DL[doc_id]
    seen = set()
    for t in q_terms:
        if t in seen:  # повтор терма в запросе не удваиваем
            continue
        seen.add(t)
        ni = DF.get(t, 0)
        if ni == 0:
            continue
        idf = math.log((len(DOCS) - ni + 0.5) / (ni + 0.5) + 1)
        # tf в документе
        tf = 0
        plist = POST.get(t)
        if plist:
            for d, f, _ in plist:
                if d == doc_id:
                    tf = f; break
        denom = tf + k1 * (1 - b + b * dl / (AVGDL or 1.0))
        score += idf * (tf * (k1 + 1) / (denom if denom else 1))
    return score

def proximity_score(q_terms: List[str], doc_id: int) -> float:
    """Приближённо: чем ближе термы друг к другу, тем выше (0..1)."""
    # собираем позиции для термов запроса
    pos_lists = []
    uniq = []
    for t in q_terms:
        if t in uniq:
            continue
        uniq.append(t)
        lst = []
        for d, _, ps in POST.get(t, []):
            if d == doc_id:
                lst = ps; break
        if lst:
            pos_lists.append(lst)
    if len(pos_lists) < 2:
        return 0.0
    # оценим минимальное окно между любыми двумя списками
    best = 1e9
    for i in range(len(pos_lists)):
        for j in range(i+1, len(pos_lists)):
            a, b = pos_lists[i], pos_lists[j]
            pa = pb = 0
            while pa < len(a) and pb < len(b):
                best = min(best, abs(a[pa]-b[pb]))
                if a[pa] < b[pb]: pa += 1
                else: pb += 1
    return 1.0 / (1.0 + best) if best < 1e9 else 0.0

def heading_hit_score(q_terms: List[str], doc: Doc) -> float:
    h = (doc.section_path or "") + " " + (doc.heading or "")
    ht = set(tok(h))
    qt = [t for t in q_terms if t not in STOP]
    if not qt:
        return 0.0
    inter = sum(1 for t in set(qt) if t in ht)
    return inter / len(set(qt))  # 0..1

def context_boosts(q_ctx, doc: Doc) -> float:
    w_head, w_role, w_law, w_topic, w_near = 0.6, 0.3, 0.4, 0.2, 0.3
    q_terms = norm_query(" ".join([x for x in [q_ctx.get("law"), q_ctx.get("role"), q_ctx.get("topic")] if x]))
    head = heading_hit_score(q_terms, doc)  # 0..1
    role = 1.0 if (q_ctx["role"] and doc.role and q_ctx["role"] in doc.role) else 0.0
    law  = 1.0 if (q_ctx["law"] and (doc.law_hint and q_ctx["law"] == doc.law_hint)) else 0.0
    topic= 1.0 if (q_ctx["topic"] and doc.topic and q_ctx["topic"][:5] in doc.topic) else 0.0
    # proximity считаем позже, здесь только веса заглушкой
    return w_head*head + w_role*role + w_law*law + w_topic*topic, w_near

def search(query: str, top_k: int = 5) -> List[Dict]:
    if not BUILT:
        build_index()
    q_ctx = parse_context(query)
    q_terms = [t for t in norm_query(query) if t not in STOP]

    # кандидаты — любые документы, где встречается хоть один терм
    cand = set()
    for t in set(q_terms):
        for d, _, _ in POST.get(t, []):
            cand.add(d)

    scored = []
    for doc_id in cand:
        base = bm25_score(q_terms, doc_id)
        if base <= 0:
            continue
        doc = DOCS[doc_id]
        ctx_boost, w_near = context_boosts(q_ctx, doc)
        prox = proximity_score(q_terms, doc_id)
        final = 1.0*base + ctx_boost + w_near*prox
        scored.append((final, doc_id, base, ctx_boost, prox))

    scored.sort(reverse=True)
    out = []
    for s, doc_id, base, ctx, prox in scored[:top_k]:
        d = DOCS[doc_id]
        out.append({
            "score": round(s, 4),
            "base": round(base, 4),
            "ctx": round(ctx, 4),
            "prox": round(prox, 4),
            "title": d.title,
            "url": d.url,
            "section": d.section_path,
            "heading": d.heading,
            "snippet": make_snippet(d.text, q_terms, max_len=500),
        })
    return out

def make_snippet(text: str, q_terms: List[str], max_len=500) -> str:
    # подсветим совпадения
    terms = sorted(set([t for t in q_terms if len(t) > 2]), key=len, reverse=True)
    s = text
    for t in terms[:10]:
        s = re.sub(rf"(?i)\b({re.escape(t)})\b", r"**\1**", s)
    # обрежем по границе предложения
    if len(s) <= max_len:
        return s
    cut = s[:max_len]
    p = cut.rfind(". ")
    return cut[:p+1] if p > 100 else cut + "…"