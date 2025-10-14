from __future__ import annotations
import os, json, uuid, hashlib
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

# ---------- Настройки ----------

EMBEDDING_MODEL_NAME = "/home/user/ru-en-RoSBERTa/"

CHUNK_SIZE = 1200         # символов (~200–350 токенов для RU)
VECTOR_SIZE = 1024
CHUNK_OVERLAP = 200       # символов
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! "]

# Если используешь cosine в векторной БД — оставь True
NORMALIZE_EMBEDDINGS = True

# ---------- Модель эмбеддингов ----------

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def make_chunks(long_text: str) -> List[str]:
    """Разбивает текст на перекрывающиеся чанки с учётом границ предложений."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS
    )
    return splitter.split_text(long_text)

PROMPT_MAP = {
    "query": "search_query",
    "passage": "search_document",
    "classification": "classification",
    "clustering": "clustering",
}

def get_embeddings_local(
    texts: Union[str, List[str]],
    mode: str = "passage",  # "query" | "passage" | ...
    convert_to_numpy: bool = True,
) -> Union[List[float], List[List[float]]]:
    """RoSBERTa с prompt_name (фоллбэк — строковый префикс)."""
    if isinstance(texts, str):
        texts_batch = [texts]
        single = True
    else:
        texts_batch = texts
        single = False

    prompt_name = PROMPT_MAP.get(mode)

    try:
        vecs = embedding_model.encode(
            texts_batch,
            prompt_name=prompt_name,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=convert_to_numpy,
            batch_size=64,
            show_progress_bar=False,
        )
    except TypeError:
        prefix = (prompt_name + ": ") if prompt_name else ""
        inputs = [prefix + t for t in texts_batch]
        vecs = embedding_model.encode(
            inputs,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=convert_to_numpy,
            batch_size=64,
            show_progress_bar=False,
        )

    if single:
        return vecs.tolist() if convert_to_numpy else vecs
    return [v.tolist() for v in vecs] if convert_to_numpy else vecs

# ---------- Qdrant конфиг ----------------------

# QDRANT_URL = "http://localhost:6333"
# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "roseltorg"
FILE_GLOB = "parsed_data.json"

def ensure_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = VECTOR_SIZE,          # 1024 для RoSBERTa
    distance: Distance = Distance.COSINE,
    recreate_on_mismatch: bool = False,
) -> None:
    """Создаёт коллекцию либо валидирует размерность/метрику."""
    assert embedding_model.get_sentence_embedding_dimension() == vector_size, \
        f"Размерность модели {embedding_model.get_sentence_embedding_dimension()} != {vector_size}"
    try:
        info = client.get_collection(collection_name)  # коллекция существует
        vp = info.config.params.vectors  # ожидаем один вектор без имён
        current_size = vp.size
        current_distance = vp.distance
        if current_size != vector_size or current_distance != distance:
            if recreate_on_mismatch:
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance),
                )
            else:
                raise ValueError(
                    f"Коллекция '{collection_name}' имеет size={current_size}, distance={current_distance}; "
                    f"ожидалось size={vector_size}, distance={distance}."
                )
    except UnexpectedResponse:
        # коллекции нет — создаём
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )


def deterministic_point_id(collection: str, doc_id: str, chunk_index: int, chunk_text: str) -> str:
    """Детерминированный UUIDv5 по (collection|doc_id|chunk_index|sha1(chunk))."""
    name = f"{collection}|{doc_id}|{chunk_index}|" + hashlib.sha1(chunk_text.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


def iter_json_records(path: Path) -> Iterable[Dict[str, Any]]:
    """Читает JSON-массив или JSONL и отдаёт словари."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            yield data
        elif isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    yield rec
    except Exception:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict):
                    yield rec


def add_documents(
        client: QdrantClient,
        collection_name: str,
        docs: Iterable[Dict[str, Any]],
        text_field: str = "text",
        id_field: Optional[str] = "url",  # используем URL как стабильный doc_id
        meta_fields: Optional[List[str]] = None,  # метаданные в payload
        upsert_batch: int = 2048,
) -> int:
    """Добавляет документы в Qdrant как чанки (идемпотентно по id)."""
    ensure_qdrant_collection(client, collection_name)
    meta_fields = meta_fields or ["title", "url"]

    points: List[PointStruct] = []
    added_chunks = 0

    for doc in docs:
        full_text = (doc.get(text_field) or "").strip()
        if not full_text:
            continue

        doc_id = (doc.get(id_field) or f"doc-{uuid.uuid4()}") if id_field else f"doc-{uuid.uuid4()}"
        chunks = make_chunks(full_text)
        vectors = get_embeddings_local(chunks, mode="passage")  # RoSBERTa + search_document:

        payload_base = {k: doc.get(k) for k in meta_fields if k in doc}
        payload_base["doc_id"] = doc_id

        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            pid = deterministic_point_id(collection_name, doc_id, i, chunk)
            payload = {
                **payload_base,
                "text": chunk,
                "chunk_index": i,
                "chunk_count": len(chunks),
            }
            points.append(PointStruct(id=pid, vector=vec, payload=payload))
            added_chunks += 1

        if len(points) >= upsert_batch:
            client.upsert(collection_name=collection_name, points=points)
            points = []

    if points:
        client.upsert(collection_name=collection_name, points=points)

    return added_chunks


def ingest_dir(
    data_dir: str,
    collection_name: str = COLLECTION_NAME,
    file_glob: str = FILE_GLOB,
) -> int:
    """Обходит все файлы в папке и заливает в Qdrant."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    total = 0
    for path in sorted(Path(data_dir).glob(file_glob)):
        records = list(iter_json_records(path))
        added = add_documents(
            client=client,
            collection_name=collection_name,
            docs=records,
            text_field="text",
            id_field="url",
            meta_fields=["title", "url"],
        )
        print(f"{path.name}: добавлено чанков {added}")
        total += added
    print(f"Готово. Всего добавлено чанков: {total}")
    return total

# ---------- Поиск (retrieve) ----------

def retrieve(
    client: QdrantClient,
    collection_name: str,
    query: str,
    top_k: int = 24,
    score_threshold: Optional[float] = None,
    filter_: Optional[Any] = None,  # qdrant_client.http.models.Filter
) -> List[Dict[str, Any]]:
    """Ищет релевантные чанки по косинусу."""
    qvec = get_embeddings_local(query, mode="query")
    results = client.search(
        collection_name=collection_name,
        query_vector=qvec,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=filter_,
        with_payload=True,
        with_vectors=False,
    )
    return [{
        "id": r.id,
        "score": r.score,
        "text": r.payload.get("text", ""),
        "payload": r.payload,
    } for r in results]

# пример запуска заливки:
# ingest_dir(r"C:\Users\famil\Desktop\json_data")

