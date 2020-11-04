import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# <Step1. 크롤링> : 크롤링으로 웹 데이터 가져오기
# 크롤링으로 웹 데이터 가져오기
# 페이지 리스트 가져오기
import requests
from bs4 import BeautifulSoup
import re

# 크롤링할 사이트 주소를 정의합니다.
source_url = "https://namu.wiki/RecentChanges"

# 사이트의 html 구조에 기반하여 크롤링을 수행합니다.
req = requests.get(source_url)
html = req.content
soup = BeautifulSoup(html, 'lxml')
contents_table = soup.find(name="table")
table_body = contents_table.find(name="tbody")
table_rows = table_body.find_all(name="tr")

# a태그의 href 속성을 리스트로 추출하여, 크롤링 할 페이지 리스트를 생성합니다.
page_url_base = "https://namu.wiki"
page_urls = []
for index in range(0, len(table_rows)):
    first_td = table_rows[index].find_all('td')[0]
    td_url = first_td.find_all('a')
    if len(td_url) > 0:
        page_url = page_url_base + td_url[0].get('href')
        if 'png' not in page_url:
            page_urls.append(page_url)

# 중복 url을 제거합니다.
page_urls = list(set(page_urls))
'''
for page in page_urls[:5]:
    print(page)
'''

# 페이지 내 텍스트 구조 확인
req = requests.get(page_urls[0])
html = req.content
soup = BeautifulSoup(html, 'lxml')
contents_table = soup.find(name="article")
title = contents_table.find_all('h1')[0]
category = contents_table.find_all('ul')[0]
content_paragraphs = contents_table.find_all(name="div", attrs={"class":"wiki-paragraph"})
content_corpus_list = []

for paragraphs in content_paragraphs:
    content_corpus_list.append(paragraphs.text)
content_corpus = "".join(content_corpus_list)

'''
print(title.text)
print("\n")
print(category.text)
print("\n")
print(content_corpus)
'''

# [나무위키 최근변경 데이터 크롤링]

# 크롤링한 데이터를 데이터 프레임으로 만들기 위해 준비합니다.
columns = ['title', 'category', 'content_text']
df = pd.DataFrame(columns=columns)

# 각 페이지별 '제목', '카테고리', '본문' 정보를 데이터 프레임으로 만듭니다.
for page_url in page_urls:

    # 사이트의 html 구조에 기반하여 크롤링을 수행합니다.
    req = requests.get(page_url)
    html = req.content
    soup = BeautifulSoup(html, 'lxml')
    contents_table = soup.find(name="article")
    title = contents_table.find_all('h1')[0]
    
    # 카테고리 정보가 없는 경우를 확인합니다.
    if len(contents_table.find_all('ul')) > 0:
        category = contents_table.find_all('ul')[0]
    else:
        category = None
        
    content_paragraphs = contents_table.find_all(name="div", attrs={"class":"wiki-paragraph"})
    content_corpus_list = []
    
    # 페이지 내 제목 정보에서 개행 문자를 제거한 뒤 추출합니다. 만약 없는 경우, 빈 문자열로 대체합니다.
    if title is not None:
        row_title = title.text.replace("\n", " ")
    else:
        row_title = ""
    
    # 페이지 내 본문 정보에서 개행 문자를 제거한 뒤 추출합니다. 만약 없는 경우, 빈 문자열로 대체합니다.
    if content_paragraphs is not None:
        for paragraphs in content_paragraphs:
            if paragraphs is not None:
                content_corpus_list.append(paragraphs.text.replace("\n", " "))
            else:
                content_corpus_list.append("")
    else:
        content_corpus_list.append("")
        
    # 페이지 내 카테고리정보에서 “분류”라는 단어와 개행 문자를 제거한 뒤 추출합니다. 만약 없는 경우, 빈 문자열로 대체합니다.
    if category is not None:
        row_category = category.text.replace("\n", " ")
    else:
        row_category = ""
    
    # 모든 정보를 하나의 데이터 프레임에 저장합니다.
    row = [row_title, row_category, "".join(content_corpus_list)]
    series = pd.Series(row, index=df.columns)
    df = df.append(series, ignore_index=True)

# 데이터 프레임을 출력합니다.
# df.head(5)


# <Step2. 추출> : 키워드 추출
#[텍스트 데이터 전처리]

# 텍스트 정제 함수: 한글 이외의 문자는 전부 제거
def text_cleaning(text):
    hangul = re.compile('[^ ㄱ-ㅣ가-힇]')
    result = hangul.sub('', text)
    return result
# print(text_cleaning(df['content_text'][0]))

# 각 피처마다 데이터 전처리 적용
df['title'] = df['title'].apply(lambda x: text_cleaning(x))
df['category'] = df['category'].apply(lambda x: text_cleaning(x))
df['content_text'] = df['content_text'].apply(lambda x: text_cleaning(x))
#print(df.head(5))

# 빈도 분석 - 명사, 혹은 형태소 단위의 문자열을 추출한 후 빈도를 분석한다
# 이를 수행하기 위해 말뭉치를 만듦
# 각 피처마다 말뭉치 생성: tolist()로 추출한 뒤 join함수를 이용해 다 묵어줌
title_corpus = ''.join(df['title'].tolist())
category_corpus = ''.join(df['category'].tolist())
content_corpus = ''.join(df['content_text'].tolist())
# print(title_corpus)

# 각 말뭉치 안에서 형태소 추출: konlpy를 사용하여 추출
from konlpy.tag import Okt
from collections import Counter

# konlpy의 형태소 분석기로 명사 단위의 키워드를 추출합니다.
nouns_tagger = Okt()
nouns = nouns_tagger.nouns(content_corpus)
count = Counter(nouns)
# print(count)

# 한 글자 키워드 제거: 의미를 가지는 한 글자 키워드는 따로 예외 처리를 해주지만 해당 과정에서는 제외
remove_char_counter = Counter({x: count[x] for x in count if len(x) > 1})
#print(remove_char_counter)

# 불용어 제외: '입니다', '그', '저' 등 관사나 접속사 등 실질적인 의미가 없고 의미적인 독립을 할 수 없는 품사 제외
# 한국어의 약식 불용어 사전을 이용해서 제외
korean_stopwords_path = 'korean_stopwords.txt'

with open(korean_stopwords_path, encoding='utf8') as f:
    stopwords = f.readlines()
stopwords = [x.strip() for x in stopwords]
#print(stopwords[:10])

# 불용어 추가
namu_wiki_stopwords = ['상위', '문서', '내용', '누설', '아래', '해당', '설명', '표기', '추가', '모든', '사용', '매우', '가장', '줄거리', '요소','상황', '편집', '틀', '경우', '때문', '모습', '정도', '이후', '사실', '생각', '인물', '이름', '년월']
for stopword in namu_wiki_stopwords:
    stopwords.append(stopword)

remove_char_counter = Counter({x: remove_char_counter[x] for x in count if x not in stopwords})
#print(remove_char_counter)

# <Step3. 시각화> : 워드 클라우드 시각화
# pip install pytagcloud pygame simplejson
# 폰트 다운로드 후 Lib\site-packages\pytagcloud\fonts 에 저장
# font.json 파일 수정:
'''
[{
      "name": "NanumGothic",
      "ttf": "NanumGothic.ttf",
      "web": "http://fonts.googleapis.com/css?family=Nanum+Gothic"
}]
'''

# pytagcloud 사용하기
import random
import pytagcloud
import webbrowser

# 가장 빈도 높은 40개 단어 선정
ranked_tags = remove_char_counter.most_common(40)
# pytagcloud로 출력할 40개 단어 입력, 단어 출력 크기는 최대 80
taglist = pytagcloud.make_tags(ranked_tags, maxsize=80)
# pytagcloud 이미지 생성
pytagcloud.create_tag_image(taglist, 'wordcloud.jpg', size=(900, 600),
fontname = 'NanumGothic', rectangular = False)

# 생성한 이미지 출력
# 주피터 노트북 상에서 출력하는 명령어인데 vs code에서 실행이 안됨
# #%% 을 붙여서 해보라길래 붙여봤는데, 잘 된다
#%%
from IPython.display import Image
Image(filename = 'wordcloud.jpg')
# %%
