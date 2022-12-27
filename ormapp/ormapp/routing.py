from django.conf.urls import url
from keystroke.wsapp import wsHandler

websocket_urlpatterns = [
    url(r'', wsHandler.as_asgi()),
]
