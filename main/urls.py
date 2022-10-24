

from django.urls import re_path as url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^response$', views.auto_response, name='auto_response'),
#     url(r'^response_api$', views.response_api, name='response_api'),
]