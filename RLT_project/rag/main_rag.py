import subprocess
from qdrant_client import QdrantClient
from qdrant_client.http import models
from normalize_query import normalise_query, TERMINS
from embed_query import get_embedding  # <-- твоя функция вынесена в отдельный файл embeddings.py

# === 1. Подключение к Qdrant ===
client = QdrantClient(host="localhost", port=6333)
collection_name = "docs"


# === 2. Вызов локальной модели GPT-OSS:20b через Ollama ===
def _call_local_gpt(prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", "gpt-oss:20b"],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"search_query: ERROR {e.stderr.strip()}"


# === 3. Поиск релевантных документов в Qdrant ===
def search_in_qdrant(query: str, top_k: int = 3):
    vector = get_embedding(query).tolist()
    hits = client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k,
    )
    return hits


# === 4. Основной пайплайн RAG ===
def rag_pipeline(user_message: str):
    # Нормализация запроса (если у тебя есть такие правила)
    normalized_query = normalise_query(user_message, TERMINS)

    # Шаг 1: Поиск кандидатов в Qdrant
    hits = search_in_qdrant(normalized_query, top_k=3)

    if not hits:
        return "Перевод на оператора"

    # Шаг 2: Собираем контекст из найденных статей
    context_parts = []
    sources = []
    for hit in hits:
        payload = hit.payload
        context_parts.append(f"{payload['title']} ({payload['url']}): {payload['text']}")
        sources.append(payload['url'])

    context = "\n\n".join(context_parts)

    # Шаг 3: Формируем prompt для LLM
    prompt = f"""
Ты — экспертная система поддержки пользователей.
У тебя есть база знаний (ниже).
Пользователь задал вопрос: "{user_message}".

База знаний:
{context}

Задачи:
1. Определи категорию запроса:
   - "ответ по работе пользователя"
   - "ответ по проблеме"
   - "ответ на термин"
2. Сформулируй понятный ответ для пользователя, основываясь на базе знаний.
3. В конце добавь строку: "Источник: {sources[0]}"
4. Если информации недостаточно, напиши: "Перевод на оператора".
"""

    # Шаг 4: Ответ от GPT-OSS
    llm_answer = _call_local_gpt(prompt)
    return llm_answer


# === 5. Пример использования ===
if __name__ == "__main__":
    test_query = "Как зарегистрироваться поставщику по 44-ФЗ?"
    answer = rag_pipeline(test_query)
    print("🤖 Ответ:\n", answer)
