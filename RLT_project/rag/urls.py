from django.urls import path
from .views import api_ask, feedback_view

urlpatterns = [
    path('ask/', api_ask, name='api_ask'),
    path('feedback/', feedback_view, name='feedback')
]
