import tweepy

# 발급 완료된 key 입력
CONSUMER_KEY = '{}'
CONSUMER_SECRET = '{}'
ACCESS_TOKEN_KEY = '{-'
ACCESS_TOKEN_SECRET = '{}'


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