from django.shortcuts import render


def chat_page(requests):
    return render(requests, 'chat/chat.html')