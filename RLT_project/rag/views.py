from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404
from django.db import transaction
from .search import build_index, search
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from chat.models import User, Message, Chat
from chat.serializers import ChatSerializer, UserSerializer, MessageSerializer

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


@csrf_exempt
@require_http_methods(["POST"])
def api_ask(request):
    try:
        data = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Невалидный JSON"}, status=400)

    question = (data.get("question") or "").strip()
    if not question:
        return JsonResponse({"error": "Пустой вопрос"}, status=400)

    # можно передавать готовые id, если создаёшь сообщения в существующем чате/для существующего пользователя
    user_id = data.get("user_id")
    chat_id = data.get("chat_id")

    try:
        with transaction.atomic():
            # 1) находим/создаём пользователя
            if user_id:
                user = get_object_or_404(User, id=user_id)
            else:
                user = User.objects.create(role="customer")

            # 2) находим/создаём чат
            if chat_id:
                chat = get_object_or_404(Chat, id=chat_id)
            else:
                chat = Chat.objects.create(user=user)

            # 3) сохраняем входящее сообщение пользователя
            in_msg_ser = MessageSerializer(data={
                "chat": str(chat.id),
                "author": str(user.id),
                "text": question,
            })
            if not in_msg_ser.is_valid():
                return JsonResponse({"errors": in_msg_ser.errors}, status=400)
            in_msg = in_msg_ser.save()

            # 4) ищем материалы
            results = search(question, top_k=3)
            citations = [{
                "title": r.get("title") or "",
                "snippet": (r.get("text") or "")[:300],
                "url": r.get("url") or "",
            } for r in results]

            # 5) создаём/находим бота и пишем ответ
            bot, _ = User.objects.get_or_create(role="llm_bot", defaults={"is_active": True})
            answer_text = "Готово. Нашли подходящие материалы."

            out_msg_ser = MessageSerializer(data={
                "chat": str(chat.id),
                "author": str(bot.id),
                "text": answer_text,
                "is_read": True,
            })
            if not out_msg_ser.is_valid():
                return JsonResponse({"errors": out_msg_ser.errors}, status=400)
            out_msg = out_msg_ser.save()

        return JsonResponse({
            "answer": answer_text,
            "citations": citations,
            "ids": {
                "user": str(user.id),
                "chat": str(chat.id),
                "question_message": str(in_msg.id),
                "answer_message": str(out_msg.id),
            }
        }, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
