from django.urls import path
from scanning import views

urlpatterns = [
    path('', views.predict_resume_category, name='predict'),

]