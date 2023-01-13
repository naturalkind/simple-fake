"""ormapp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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
from django.urls import path, re_path
from django.conf import settings
from django.conf.urls.static import static

from keystroke import views as keystroke

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', keystroke.mainpage),
    path(r'registration/', keystroke.registration),
    path(r'registrationend/', keystroke.registrationend),
    path(r'login/', keystroke.login),
    path(r'keystroke/', keystroke.keystroke),
    re_path(r'^data/(?P<post>\d+)/$', keystroke.post), # страница материала
    re_path(r'^user_page/(?P<user_id>\d+)/$', keystroke.user_page), # страница пользователя
    path(r'logout/', keystroke.quit),
    path(r'gettext/', keystroke.gettext),
    path(r'alldata/', keystroke.alldata),
    path(r'cratealldata/', keystroke.cratealldata),
    
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
