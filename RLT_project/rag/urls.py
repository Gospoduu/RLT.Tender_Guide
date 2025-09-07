from django.urls import path
from .views import api_ask

urlpatterns = [
    path('api/', api_ask, name='api_ask'),
]

