from __future__ import annotations
import re, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Iterable, Dict, Tuple, Optional

DATA_DIR = Path(__file__).parent / "data"
SRC_JSON = DATA_DIR / "parsed_data.json"
OUT_JSONL = DATA_DIR / "chunks.jsonl"

# ==== очистка ====
SP = re.compile(r"[ \t]+")
NL = re.compile(r"\r\n?")


def clean_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\u00a0"," ").replace("\ufeff"," ")
    s = NL.sub("\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return SP.sub(" ", s).strip()

# ==== детект структурных блоков ====
LIST_RX = re.compile(r"^\s*([-*•]|[0-9]+\)|[0-9]+\.)\s+")
HEADING_RX = re.compile(
    r"""^(
        \s*#{1,6}\s+.*|                        # Markdown заголовки
        \s*(раздел|глава|статья)\s+\d+[^\n]*|  # Раздел/Глава/Статья N
        \s*\d+(\.\d+){0,3}\)?\s+[^\n]+$|       # 1. / 1.1. / 1.1.1)
        \s*[А-ЯЁ][А-ЯЁ0-9\s\-,:]{8,}$          # ALL CAPS блоки
    )""",
    re.I | re.X
)
QUESTION_RX = re.compile(r".+\?\s*$")          # FAQ-вопросы как подзаголовки
ANCHOR_RX = re.compile(r"^(Определение|Порядок|Сроки|Требования|Частые ошибки|Заключение)\b", re.I)
STEP_RX = re.compile(r"^(Этап|Шаг|Вариант)\s+\d+", re.I)

SENT_SPLIT = re.compile(r"(?<=[\.\?\!…])\s+")

def paragraphize(text: str) -> List[str]:
    lines = text.split("\n")
    paras, buf = [], []
    for ln in lines:
        if LIST_RX.match(ln.strip()):
            buf.append(ln.strip()); continue
        if ln.strip()=="":
            if buf: paras.append(" ".join(buf)); buf=[]
            continue
        if buf: paras.append(" ".join(buf)); buf=[]
        paras.append(ln.strip())
    if buf: paras.append(" ".join(buf))
    # убрать очень короткие
    return [p for p in paras if len(p.replace(" ","")) > 1]

def split_into_subtopics(paras: List[str]) -> List[Tuple[str, List[str]]]:
    # === Делим статью на «подтемы»: заголовки/вопросы/якоря/этапы открывают новый блок. ===
    out: List[Tuple[str, List[str]]] = []
    head = "Без подзаголовка"
    bucket: List[str] = []
    for p in paras:
        if HEADING_RX.match(p) or QUESTION_RX.match(p) or ANCHOR_RX.match(p) or STEP_RX.match(p):
            # закрыть предыдущий блок
            if bucket:
                out.append((head, bucket)); bucket = []
            head = re.sub(r"^\s*#+\s*", "", p).strip()
            continue
        bucket.append(p)
    if bucket:
        out.append((head, bucket))
    return out

def pack_chunks(title: str, heading: str, paras: List[str],
                max_chars: int = 1100, overlap_sentences: int = 2) -> List[str]:
    # === Упаковка абзацев в чанки. Списки уже склеены; если длинно — режем по предложениям.===
    joined = "\n\n".join(paras).strip()
    if not joined: return []
    # короткие FAQ-ответы можно отдать одним чанком без резки
    if len(joined) <= max_chars:
        return [f"{title} › {heading}: {joined}"]

    sentences = SENT_SPLIT.split(joined)
    # позиции предложений, чтобы корректно брать overlap
    pos=[]; cur=0
    for s in sentences:
        i = joined.find(s, cur)
        if i < 0: i = cur
        pos.append((i, i+len(s))); cur = i+len(s)

    chunks=[]; i=0
    while i < len(sentences):
        start_i = i
        start_char = pos[start_i][0]
        length = 0; j=i
        while j < len(sentences) and length + len(sentences[j]) <= max_chars:
            length += len(sentences[j]); j += 1
        end_char = pos[j-1][1]
        text = joined[start_char:end_char].strip()
        if len(text) >= 180:  # избегаем мелочи
            chunks.append(f"{title} › {heading}: {text}")
        # шаг с перекрытием
        i = max(j - overlap_sentences, j) if j > i else j + 1
    return chunks

# ==== фасеты (минимум для контекста) ====
LAW_RX = re.compile(r"\b(44\s*-\s*фз|223\s*-\s*фз|615\s*ПП\s*РФ)\b", re.I)
ROLE_RX = re.compile(r"\b(поставщик|заказчик|комиссия|оператор|покупатель|продавец)\b", re.I)
TOPIC_RX = re.compile(r"\b(регистрация|аккредитация|змо|запрос предложений|жалоб|нмцк|срок|требовани|штраф|договор|контракт)\w*\b", re.I)


def extract_facets(text: str) -> Dict[str, Optional[str]]:
    law = (LAW_RX.search(text) or [None])[0]
    role = (ROLE_RX.search(text) or [None])[0]
    topic = (TOPIC_RX.search(text) or [None])[0]
    return {
        "law_hint": law.upper().replace(" ", "") if law else None,
        "role": role.lower() if role else None,
        "topic": (topic.lower() if topic else None)
    }


# ==== модель чанка ====
@dataclass
class Chunk:
    title: str
    url: str
    heading: str
    section_path: str
    text: str
    law_hint: Optional[str] = None
    role: Optional[str] = None
    topic: Optional[str] = None


def make_chunks_for_item(title: str, url: str, text: str,
                         max_chars=1100, overlap_sentences=2) -> List[Chunk]:
    title = (title or "Без названия").strip()
    url = (url or "").strip()
    body = clean_text(text)
    paras = paragraphize(body)
    subtopics = split_into_subtopics(paras) or [("Без подзаголовка", paras)]

    out: List[Chunk] = []
    for heading, block in subtopics:
        heading = heading.strip() or "Без подзаголовка"
        parts = pack_chunks(title, heading, block, max_chars=max_chars,
                            overlap_sentences=overlap_sentences)
        for p in parts:
            # фасеты берём из «чистой» части без префикса
            facet_src = p.split(":", 1)[-1]
            facets = extract_facets(facet_src)
            out.append(Chunk(
                title=title,
                url=url,
                heading=heading,
                section_path=f"{title} › {heading}",
                text=p.split(": ", 1)[-1],
                law_hint=facets["law_hint"],
                role=facets["role"],
                topic=facets["topic"],
            ))
    return out


def build_all_chunks(max_chars=1100, overlap_sentences=2) -> int:
    data = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    count = 0
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for it in data:
            title, url, text = it.get("title",""), it.get("url",""), it.get("text","")
            chunks = make_chunks_for_item(title, url, text, max_chars, overlap_sentences)
            for ch in chunks:
                f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")
                count += 1
    return count