from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_price, name='home'),  # 👈 Connect to frontend.html
]
