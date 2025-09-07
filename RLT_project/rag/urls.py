from django.urls import path
from .views import api_ask

urlpatterns = [
    path('ask/', api_ask, name='api_ask'), 
]
