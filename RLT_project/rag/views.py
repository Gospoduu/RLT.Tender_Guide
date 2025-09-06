from django.shortcuts import render
from .search import build_index, search
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json


def answer_question(question: str, top_k: int = 3) -> dict:
    """
    Принимает вопрос, возвращает релевантные документы.
    Позже сюда добавим генерацию ответа LLM.
    """
    if not question.strip():
        return {"error": "Вопрос пустой."}

    results = search(question, top_k=top_k)

    return {
        "question": question,
        "results": results
    }


@csrf_exempt  # временно отключаем CSRF, пока нет защиты через токен
def api_ask(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            question = data.get("question", "").strip()

            if not question:
                return JsonResponse({"error": "Пустой вопрос"}, status=400)

            results = search(question, top_k=3)

            # для каждого результата добавляем поля title, snippet и ссылку
            citations = [{
                "title": r["title"],
                "snippet": r["text"][:300],  # можно обрезать
                "url": r["url"]
            } for r in results]

            return JsonResponse({
                "answer": "Готово. Нашли подходящие материалы:",
                "citations": citations
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Метод не поддерживается"}, status=405)