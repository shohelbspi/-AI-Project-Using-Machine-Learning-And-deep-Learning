from django.urls import path
from app import views

urlpatterns = [
        path('predict/', views.emotion_view, name='predict'),
   
]
