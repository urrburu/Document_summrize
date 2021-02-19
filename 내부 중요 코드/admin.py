from django.contrib import admin
from .models import clip
# Register your models here.
class clipAdmin(admin.ModelAdmin):

    list_display = ['contents','id','title' ]
    #관리자페이지에서 볼 수 있는 것은 글내용, 글번호, 글제목으로 제한한다.
    ordering = ['-id']
    #이를 업데이트된 순서로 나열한다.

admin.site.register(clip, clipAdmin)
