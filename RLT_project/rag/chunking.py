import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent / "data"
IN_FILE = DATA_DIR / "parsed_data.json"
OUT_FILE = DATA_DIR / "chunks.jsonl"

def build_all_chunks(chunk_size=1000, chunk_overlap=200):
    with open(IN_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []

    for article in articles:
        meta = {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
        }

        text = article.get("text", "").strip()
        if not text:
            continue

        for chunk in splitter.split_text(text):
            all_chunks.append({**meta, "text": chunk})

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    return len(all_chunks), OUT_FILE