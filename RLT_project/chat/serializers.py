from rest_framework import serializers
from .models import User, Message, Chat
from django.db import models

def create_auto_serializer(model_class, input_fields='__all__', read_only_fields=None):
    """
    Создает автоматический сериализатор для модели Django

    Args:
        model_class: Класс модели Django
        input_fields: Список полей или '__all__'
        read_only_fields: Поля только для чтения

    Returns:
        Динамически созданный сериализатор
    """

    # Базовые read_only поля
    base_read_only = ['id', 'created_at']
    if read_only_fields:
        if isinstance(read_only_fields, list):
            base_read_only.extend(read_only_fields)
        else:
            base_read_only.append(read_only_fields)

    # Автоматически добавляем display поля для choices
    serializer_attrs = {}
    if hasattr(model_class, '_meta'):
        for field in model_class._meta.fields:
            if hasattr(field, 'choices') and field.choices:
                field_name = f"{field.name}_display"
                serializer_attrs[field_name] = serializers.CharField(
                    source=f'get_{field.name}_display',
                    read_only=True
                )

    # Создаем Meta класс
    meta_attrs = {
        'model': model_class,
        'fields': input_fields,
        'read_only_fields': base_read_only,
    }

    # Добавляем Meta класс в атрибуты
    serializer_attrs['Meta'] = type('Meta', (object,), meta_attrs)

    # Создаем класс сериализатора
    serializer_class = type(
        f'Auto{model_class.__name__}Serializer',
        (serializers.ModelSerializer,),
        serializer_attrs
    )

    return serializer_class

UserSerializer = create_auto_serializer(
    User,
    fields=['id', 'role', 'is_active', 'created_at']
)

ChatSerializer = create_auto_serializer(
    Chat,
    fields=['id', 'user', 'assigned_to', 'created_at']
)

MessageSerializer = create_auto_serializer(
    Message,
    fields=['id', 'chat', 'author', 'text', 'created_at', 'is_read']
)