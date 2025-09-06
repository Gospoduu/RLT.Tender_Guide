from django.db import models
import uuid
# Create your models here.

class User(models.Model):
    USER_ROLES = [
        ('customer', 'Клиент'),
        ('support_staff', 'Сотрудник поддержки'),
        ('llm_bot', 'ИИ-ассистент'),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    role = models.CharField(max_length=20, choices=USER_ROLES, default='customer')
    is_active = models.BooleanField(default=True)
class Chat(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chats')
    created_at = models.DateTimeField(auto_now_add=True)
    assigned_to = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True,
                                    related_name='assigned_chats')
class Message(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name='messages')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='messages')
    text = models.TextField()  # Только текстовое поле
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    is_read = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.author.id}: {self.text[:50]}..."

