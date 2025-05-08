from django.urls import path
from recommendation import views

urlpatterns = [
    path('', views.remmendation, name='recommend'),
]
