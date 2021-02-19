from django.urls import path
from django.views.generic.detail import DetailView
from rest_framework.urlpatterns import format_suffix_patterns
from django.conf.urls import include, url

from .views import clip_list, clip_new, PostList
from .models import clip
app_name = 'clip'

urlpatterns = [
    path('',clip_list, name='clip_list'),
    path('upload/', clip_new, name='clip_new'),
    path('clip/tags/<tag>', clip_list, name='clip_search'),
    path('post/', PostList.as_view())
    
]
