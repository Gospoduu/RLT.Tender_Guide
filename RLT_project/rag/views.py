from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.shortcuts import get_object_or_404
from django.db import transaction
import json

from chat.models import User, Message, Chat
from chat.serializers import MessageSerializer
from .main_rag import rag_pipeline  # твой RAG пайплайн


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

            # 4) получаем ответ из RAG пайплайна
            answer = rag_pipeline(question)

            # 4.1) обрезаем всё до "...done thinking" включительно
            marker = "...done thinking."
            if marker in answer:
                answer = answer.split(marker, 1)[1].strip()

            # 5) разбираем источники
            sources = []
            if "Источник:" in answer:
                parts = answer.split("Источник:")
                answer_text = parts[0].strip()
                sources = [parts[1].strip()]
            else:
                answer_text = answer

            # 6) создаём сообщение от бота
            bot, _ = User.objects.get_or_create(role="llm_bot", defaults={"is_active": True})
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
            "citations": sources,
            "ids": {
                "user": str(user.id),
                "chat": str(chat.id),
                "question_message": str(in_msg.id),
                "answer_message": str(out_msg.id),
            }
        }, status=200)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
def feedback_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Метод не поддерживается'}, status=405)

    try:
        data = json.loads(request.body)
        fb_type = data.get('type')
        question = data.get('question')
        answer = data.get('answer')

        if fb_type not in ['like', 'dislike']:
            return JsonResponse({'error': 'Некорректный тип оценки'}, status=400)

        # 🔧 Можно заменить на сохранение в БД или отправку в лог
        with open('feedback_log.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'type': fb_type,
                'question': question,
                'answer': answer,
            }, ensure_ascii=False) + '\n')

        return JsonResponse({'ok': True})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
