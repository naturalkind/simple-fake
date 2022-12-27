from django.conf.urls import url
from keystroke.wsapp import B_Handler

websocket_urlpatterns = [
    url(r'wall/$', B_Handler.as_asgi()),
]
