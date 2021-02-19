from django.shortcuts import render
from django.template.context_processors import request
from django.contrib import messages
from .models import clip,Tag #모델중에 clip과 tag를 갖고온다
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.db.models import Count
from .forms import ClipPostForm #clip을 폼의 형식을 이용해 입력을 받는다.
from django.core.paginator import Paginator,EmptyPage
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser

from rest_framework.renderers import JSONRenderer
from rest_framework.views import APIView
from rest_framework.response import Response

from .serializer import clipSerializer
from .textrank import TextRank

def check(request):
    clips = clip.objects.get(clip.summary == None)
    for clip in clips:
        tr = TextRank(clip.contents)
        clip.summary = tr.summarize()#요약문 추출
        clip.title = tr.summarize()[0] #제목추출
        tag_list = tr.keywords()#핵심단어 추출
        tag_list.insert(0,'hashtag : ')
        clip.tag_field = " #".join(tag_list)
        clip.save()

def clip_list(request, tag=None):
    tag_all = Tag.objects.annotate(num_post=Count('clip'))
    

    q = request.GET.get('q','')
    if q:
        tag=q
    if tag:
        clips = clip.objects.filter(tag_set__name__iexact=tag).prefetch_related('tag_set')
        # 처음은 태그를통한 검색으로 이 함수를 작동시켰는지 확인
    else:
        clips = clip.objects.prefetch_related('tag_set').all()
        #검색으로 찾지 않았다면 전체 문서를 다 불러옴
        
    if request.method == 'POST':
        #POST유형으로 요청한다
        tag = request.POST.get('tag')
        tag_clean = ''.join(e for e in tag if e.isalnum())  # 특수문자 삭제
        return redirect('clip:clip_search', tag_clean)
    paginator = Paginator(clips,5)#페이지네이터를 이용해 글 5개씩 끊어줌
    page = request.GET.get('page')
    clips = paginator.get_page(page)
    return render(request,'clip/list.html', {'clips':clips, 'tag':tag, tag_all:tag_all })
    #정해진 탬플릿으로 이를 보내고 클립, 태그를 연동시켜줌

def clip_new(request):
    #새 클립 생성에 필요한 함수
    if request.method == 'POST':
        form = ClipPostForm(request.POST)
        #정해진 폼을 통해서 입력을 받는다.
        if form.is_valid():
            #form에 대해 타당성검사를 한다.
            clip = form.save(commit=False)
            #클립폼을 먼저 저장한다. 장고는 폼이라는 형식을 통해 입력을 받게된다. 
            #입력의 정형화를 위해 유리한 방법이다.
            tr = TextRank(clip.contents)
            clip.summary = tr.summarize()#요약문 추출
            clip.title = tr.summarize()[0] #제목추출
            tag_list = tr.keywords()#핵심단어 추출
            tag_list.insert(0,'hashtag : ')
            clip.tag_field = " #".join(tag_list)
            #이 과정에서 제목, 요약문, 태그필드는 문장들을 summarize함수로 보내서 함수에서 바꿔서 저장한다.
            clip.save()
            clip.tag_save()

            return redirect('clip:clip_list')
            #끝났다면 cliplist함수를 다시 실행시켜서 원래의 리스트를 다시 본다
    else:
        form = ClipPostForm()
    return render(request, 'clip/upload.html', {'form': form})

class PostList(APIView):

    def post(self, request, format=None):
        serializer = clipSerializer(data=request.data)
        tr = TextRank(clip.contents)
        clip.summary = tr.summarize()#요약문 추출
        clip.title = tr.summarize()[0] #제목추출
        tag_list = tr.keywords()#핵심단어 추출
        tag_list.insert(0,'hashtag : ')
        clip.tag_field = " #".join(tag_list)
        #이 과정에서 제목, 요약문, 태그필드는 문장들을 summarize함수로 보내서 함수에서 바꿔서 저장한다.
        clip.save()
        clip.tag_save()
        
        if serializer.is_valid():
            clip = serializer.save()
            
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)

    def get(self, request, fromat=None):
        queryset = clip.objects.all()
        serializer = clipSerializer(queryset, many=True)
        return Response(serializer.data)