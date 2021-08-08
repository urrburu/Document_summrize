### C. SERVER
#### ㄱ. 사용한 프레임워크와 모듈, 지원 프로그램
##### 서버구성에 사용
```
서버 프레임워크  : Django 3.0.5
API call 구성 : Django Rest Framework 1.2.9
```
##### 배포작업에 이용
```
uWSGI 2.0
Nginx 1.14
AWS EC2 
```
#### ㄴ. 중간보고서에서의 변경점
##### ㄴ-1 API call 구현
##### ㄴ-2 TextRank 알고리즘과의 연동
#### ㄷ. 배포
##### ㄷ-1 uWSGI
##### ㄷ-2 Nginx
##### ㄷ-3 nginx와 uWSGI의 연동

#### ㄹ. 결과화면 
##### ㄹ-1 사용자 화면
##### ㄹ-2 관리자 화면
```
uwsgi \ 
--http :8080 \
--home /home/ubuntu/.pyenv/versions/summary \
--chdir /srv/project/andproject \
-w config.wsgi
```
