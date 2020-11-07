import tweepy

# 발급 완료된 key 입력
CONSUMER_KEY = '{H4wC5Z6EgaG9PJv8HluFB8LDp}'
CONSUMER_SECRET = '{jlb1xS9XY2hwcxIK0ty4IvHUOlz24lxWhXMMuG1LIQDYbRFUs5}'
ACCESS_TOKEN_KEY = '{1324072650872860672-w3wCYAcCspBPt0khdyUw6AYMiQohnc}'
ACCESS_TOKEN_SECRET = '{ZANrxMuoeslcWFKHc4ZOsHs377DxHHpCUB3VfBEBrVFeY}'


# 개인정보 인증 요청하는 Handler
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)

# 인증 요청
auth.set_access_token(ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)

# twitter API 사용하기 위한 준비
api = tweepy.API(auth)

# twitter API 를 사용하여 'son heung-min' 포함된 트윗 크롤링
# entities에서 'user_mentions', 'hashtags'추출
keyword = 'son heung-min'
tweets = api.search(keyword)
for tweet in tweets:
    print(tweet.entities['user_mentions'])
    print(tweet.entities['hashtags'])
    print(tweet.text)