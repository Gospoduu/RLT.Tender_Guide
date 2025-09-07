import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from .normalize_query import normalise_query, TERMINS
import warnings

# Подавляем предупреждения о неинициализированных весах
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized")

# Загрузка модели и токенизатора
model_path = "/home/user/ru-en-RoSBERTa/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model.eval()

# Функция для получения эмбеддинга из текста
def get_embedding(text, remove_prefix=True):
    # Извлекаем текст после "search_query: " если нужно
    if remove_prefix and text.startswith("search_query:"):
        text = text[len("search_query:"):].strip()
    
    # Токенизация
    inputs = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        return_tensors="pt", 
        max_length=512
    )
    
    # Получение эмбеддингов
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Используем эмбеддинг [CLS] токена как представление предложения
    embedding = outputs.last_hidden_state[:, 0, :]
    
    # Нормализуем эмбеддинг
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    
    return embedding.cpu().numpy().flatten()

# Функция для пакетной обработки запросов
def process_queries(queries, termins=TERMINS):
    results = []
    
    for query in queries:
        # Нормализация запроса
        normalized_query = normalise_query(query, termins)
        
        # Получение эмбеддинга
        embedding = get_embedding(normalized_query)
        
        results.append({
            "original_query": query,
            "normalized_query": normalized_query,
            "embedding": embedding,
            "embedding_shape": embedding.shape,
            "embedding_norm": np.linalg.norm(embedding)
        })
    
    return results

# Пример использования
if __name__ == "__main__":
    # Примеры запросов
    test_queries = [
        "Как оформить ЭДО для 44 фз и использовать ЛК оператора?",
        "Какие требования к электронной подписи по 223-ФЗ?",
        "Как подать жалобу в ФАС по 44-ФЗ?",
        "Инструкция по работе с ЕИС для начинающих"
    ]
    
    # Обработка запросов
    results = process_queries(test_queries)
    
    # Вывод результатов
    for i, result in enumerate(results):
        print(f"\n--- Запрос {i+1} ---")
        print(f"Оригинал: {result['original_query']}")
        print(f"Нормализованный: {result['normalized_query']}")
        print(f"Размерность эмбеддинга: {result['embedding_shape']}")
        print(f"Норма эмбеддинга: {result['embedding_norm']:.4f}")
    
    # Пример сравнения эмбеддингов
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("\n--- Сравнение эмбеддингов ---")
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            sim = cosine_similarity(
                [results[i]['embedding']], 
                [results[j]['embedding']]
            )[0][0]
            print(f"Схожесть запроса {i+1} и {j+1}: {sim:.4f}")
