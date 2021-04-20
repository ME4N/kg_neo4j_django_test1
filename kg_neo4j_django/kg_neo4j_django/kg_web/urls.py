from django.contrib import admin
from django.urls import path
from django.urls import include, path
from . import views

from django.conf.urls import url

urlpatterns = [

    # url(r'^qindex', include('kg_web.urls')),
    # url(r'^click_word', include('kg_web.urls')),
    # url(r'^ner', include('kg_web.urls')),
    # url(r'^rela', include('kg_web.urls')),

    path('qindex', views.qindex,name='qindex'),
    path('click_word', views.click_word,name='click_word'),
    path('ner', views.ner,name='ner'),
    path('relation', views.relation,name='relation'),
    path('add_entity',views.add_entity,name='add_entity'),
    path('del_relation',views.del_relation,name='del_relation')
]