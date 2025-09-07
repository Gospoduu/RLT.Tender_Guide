from rest_framework import serializers
from .models import User, Chat, Message
def create_auto_serializer(model_class, new_fields='__all__', read_only_fields=None):
    base_read_only = ['id', 'created_at']
    if read_only_fields:
        base_read_only.extend(read_only_fields if isinstance(read_only_fields, list) else [read_only_fields])

    serializer_attrs = {}
    extra_display_fields = []

    # объявляем *_display и запоминаем их имена
    for field in model_class._meta.fields:
        if getattr(field, 'choices', None):
            name = f"{field.name}_display"
            serializer_attrs[name] = serializers.CharField(
                source=f'get_{field.name}_display', read_only=True
            )
            extra_display_fields.append(name)

    # если поля заданы явно, докинем *_display, чтобы DRF не ругался
    if new_fields != '__all__':
        new_fields = list(new_fields) + [f for f in extra_display_fields if f not in new_fields]

    class Meta:
        model = model_class
        fields = new_fields
        read_only_fields = base_read_only

    serializer_attrs['Meta'] = Meta
    return type(f'Auto{model_class.__name__}Serializer', (serializers.ModelSerializer,), serializer_attrs)

UserSerializer = create_auto_serializer(
    User,
    new_fields=['id', 'role', 'is_active', 'created_at']  # включаем id и created_at
)

ChatSerializer = create_auto_serializer(
    Chat,
    new_fields=['id', 'user', 'assigned_to', 'created_at']
)

MessageSerializer = create_auto_serializer(
    Message,
    new_fields=['id', 'chat', 'author', 'text', 'is_read', 'created_at']
)