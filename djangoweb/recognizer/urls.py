from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),             # GET -> zobrazí HTML stránku
    path('recognize/', views.recognize, name='recognize'),  # POST -> spracuje obrázok
]