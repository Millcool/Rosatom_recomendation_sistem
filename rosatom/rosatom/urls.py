"""rosatom URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from reviews.views import review
from reviews.views import upload, upload_success, generate_pdf


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', review),
    path('review/', review, name='review'),
    path('upload/', upload, name='upload'),
    path('upload_success/', upload_success, name='upload_success'),
    path('generate_pdf/', generate_pdf, name='generate_pdf'),

]
