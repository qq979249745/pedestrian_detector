from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter

from django.conf.urls import url

from app.consumer import Consumer

application = ProtocolTypeRouter({
    # (http->django views is added by default)
    'websocket': AuthMiddlewareStack(
        URLRouter(
            [
                url(r'^path/', Consumer, ),
            ]
        )
    ),
})
