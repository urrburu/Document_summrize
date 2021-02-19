from django.db import models
from django.conf import settings
from django.shortcuts import reverse
import re

class Tag(models.Model):
    name = models.CharField(max_length=140, unique=True)

    def __str__(self):
        return self.name
#해쉬태그 모델, 최장길이는 140자
class clip(models.Model):
    #글 모델
    contents = models.TextField(null=True) # 본문내용
    tag_set = models.ManyToManyField('Tag', blank=True) # 해쉬태그 모음
   
    id = models.AutoField(auto_created=True, primary_key=True, serialize=False) # 고유글 넘버
    title = models.CharField(max_length=1000) # 글 제목 1000자 제한
    summary = models.TextField(null=True) # 요약문
    tag_field = models.TextField(null=True) # 해쉬태그 표시용
   
    
    class Meta:
        ordering = ['-id']

    def tag_save(self):
        tags = re.findall(r'#(\w+)\b', self.tag_field)
        #태그를 모델에서 처리하는 함수
        if not tags:
            return

        for t in tags:
            tag, tag_created = Tag.objects.get_or_create(name=t)
            self.tag_set.add(tag)  # NOTE: ManyToManyField 에 인스턴스 추가
