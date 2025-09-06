import json
import re
from pathlib import Path
from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path(__file__).resolve().parent / "data" / "chunks.jsonl"

def tokenize(text):
    """Простая токенизация: нижний регистр, без пунктуации"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def load_chunks(path=CHUNKS_PATH):
    """Загружает чанки и токенизирует их"""
    chunks = []
    tokenized = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            chunks.append(obj)
            tokenized.append(tokenize(obj["text"]))

    return chunks, tokenized

# При загрузке: читаем данные и строим индекс
CHUNKS, TOKENS = load_chunks()
BM25 = BM25Okapi(TOKENS)

def search(query, top_k=3):
    """Ищет наиболее релевантные чанки по запросу"""
    query_tokens = tokenize(query)
    scores = BM25.get_scores(query_tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []

    for i in top_indices:
        ch = CHUNKS[i]
        results.append({
            "score": round(scores[i], 4),
            "title": ch.get("title", ""),
            "url": ch.get("url", ""),
            "text": ch["text"][:500] + "..." if len(ch["text"]) > 500 else ch["text"]
        })

    return results

def build_index(force=False):
    """Перестроить индекс (опционально)"""
    global CHUNKS, TOKENS, BM25
    if force:
        CHUNKS, TOKENS = load_chunks()
        BM25 = BM25Okapi(TOKENS)
    return f"[RAG] index: docs={len(CHUNKS)}, avgdl={BM25.avgdl:.1f}, terms={len(BM25.doc_freqs)}"