# STEP 1 탐색: 프로야구 연봉 데이터 살펴보기

#%%matploitlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

picher_file_path = './sourceData/picher_stats_2017.csv'
batter_file_path = './sourceData/batter_stats_2017.csv'

picher = pd.read_csv(picher_file_path)
batter = pd.read_csv(batter_file_path)

picher.columns

picher.head()

# %%
print(picher.shape)
# %%
# 예측할 대상인 연봉에 대한 정보
picher['연봉(2018)'].describe()
# %%
# 2018 연봉 분포
picher['연봉(2018)'].hist(bins=100)
# %%
# 연봉의 상자 그림
picher.boxplot(column=['연봉(2018)'])
# %%
# 회귀 분석에 사용할 피처 살펴보기
picher_features_df = picher[['승', '패', '세', '홀드', '블론', '경기', '선발', '이닝', '삼진/9', '볼넷/9', 'BABIP', 'LOB%', 'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR', '연봉(2018)', '연봉(2017)']]

# 피처 각각에 대한 히스토그램을 출력
def plot_hist_each_column(df):
    plt.rcParams['figure.figsize'] = [20, 16]
    fig = plt.figure(1)
    # df의 열 개수 만큼의 subplot을 출력
    for i in range(len(df.columns)):
        ax = fig.add_subplot(5, 5, i+1)
        plt.hist(df[df.columns[i]], bins=50)
        ax.set_title(df.columns[i])
    plt.show()

plot_hist_each_column(picher_features_df)
# %%
# STEP 2 예측: 투수의 연봉 예측하기

# 판다스 형태로 정의된 데이터 출력 시, scientific-notation이 아닌 float 모양으로 출력
pd.options.mode.chained_assignment = None

# 피처 각각에 대한 스케일링을 수행하는 함수 정의
def standard_scaling(df, scale_columns):
    for col in scale_columns:
        series_mean = df[col].mean()
        series_std = df[col].std()
        df[col] = df[col].apply(lambda x: (x-series_mean)/series_std)
    return df

# 피처 각각에 대한 스케일링을 수행
scale_columns = ['승', '패', '세', '홀드', '블론', '경기', '선발', '이닝', '삼진/9', '볼넷/9', 'BABIP', 'LOB%', 'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR', '연봉(2018)', '연봉(2017)']
picher_df = standard_scaling(picher, scale_columns)
picher_df = picher_df.rename(columns ={'연봉(2018)': 'y'})
picher_df.head()
# %%
# 피처들의 단위 맞춰주기: 윈-핫  인코딩
# 팀명 피처를 윈-핫 인코딩으로 변환
team_encoding = pd.get_dummies(picher_df['팀명'])
picher_df = picher_df.drop('팀명', axis=1)
picher_df = picher_df.join(team_encoding)
team_encoding.head(5)
# %%
# 회귀 분석을 위한 학습, 테스트 데이터셋 분리

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# 학습 데이터와 테스트 데이터로 분리합니다.
X = picher_df[picher_df.columns.difference(['선수명', 'y'])]
y = picher_df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)

# 회귀 분석 계수를 학습합니다 (회귀 모델 학습)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# 학습된 계수를 출력합니다.
print(lr.coef_)
# %%
# STEP 3 평가: 예측 모델 평가하기
import statsmodels.api as sm

#statsmodel 라이브러리로 회귀 분석 수행
X_train = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train).fit()
model.summary()
# %%
# 한글 출력을 위한 사전 설정
import matplotlib.pyplot as plt
import matplotlib as mpl
#set(sorted([f.name for f in mpl.font_manager.fontManager.ttflist])) # 현재 OS 내에 설치된 폰트를 확인합니다.
mpl.rc('font', family='NanumGothicOTF')
plt.rcParams['figure.figsize'] = [20, 16]

# 회귀 계수를 리스트로 반환
coefs = model.params.tolist()
coefs_series = pd.Series(coefs)

# 변수명을 리스트로 반환
x_lables = model.params.index.tolist()

# 회구 계수를 출력
ax = coefs_series.plot(kind = 'bar')
ax.set_title('feature_coef_graph')
ax.set_xlabel('x_features')
ax.set_ylabel('coef')
ax.set_xticklabels(x_lables)
# %%
# 예측 모델의 평가하기
# 학습 데이터와 테스트 데이터로 분리
x = picher_df[picher_df.columns.difference(['선수명', 'y'])]
y = picher_df['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=19)

# 회귀 분석 모델 학습
lr = linear_model.LinearRegression()
model = lr.fit(x_train, y_train)

# 회귀 분석 모델 평가
# 실행 결과 각 값은 학습 데이터셋과 테스트 데이터셋에 대한 평가 점수
# 두 점수는 최대한 벌어지지 않는것이 좋음
# 만약 학습 점수가 테스트 점수에 비해 높다면 과적합
# 즉, 모의고사에만 특화된 공부를 한 나머지 실제 시험에서 새로운 유형에 적응하지 못한 것
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))
# %%
# 회귀 모델 평가 RMSE score라는 지표 활용할 수도 있음
# 값이 높으면 높을수록 예측이 부정확하다는 것을 의미
# 회귀 분석 모델을 평가합니다.
y_predictions = lr.predict(X_train)
print(sqrt(mean_squared_error(y_train, y_predictions))) # train RMSE score를 출력합니다.
y_predictions = lr.predict(X_test)
print(sqrt(mean_squared_error(y_test, y_predictions))) # test RMSE score를 출력합니다.
# %%
import seaborn as sns

# 피처간의 상관계수 행렬을 계산합니다.
corr = picher_df[scale_columns].corr(method='pearson')
show_cols = ['win', 'lose', 'save', 'hold', 'blon', 'match', 'start', 
             'inning', 'strike3', 'ball4', 'homerun', 'BABIP', 'LOB', 
             'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR', '2017']

# corr 행렬 히트맵을 시각화합니다.
plt.rc('font', family='NanumGothicOTF')
sns.set(font_scale=1.5)
hm = sns.heatmap(corr.values,
            cbar=True,
            annot=True, 
            square=True,
            fmt='.2f',
            annot_kws={'size': 15},
            yticklabels=show_cols,
            xticklabels=show_cols)

plt.tight_layout()
plt.show()
# %%
