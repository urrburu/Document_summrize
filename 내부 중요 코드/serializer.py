from rest_framework import serializers 
from .models import clip
class clipSerializer(serializers.ModelSerializer): 
    class Meta: 
        model =  clip# 모델 설정 
        fields = ['contents'] # 필드 설정
    