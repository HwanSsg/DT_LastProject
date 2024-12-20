import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle
from Datapreprocessor import DataPreprocessor
#데이터 불러오기
df = pd.read_csv('data\ki.csv')
# 데이터 준비
data = pd.DataFrame(df)
#데이터 전처리
# 'Time' 열을 분 단위로 변환하는 함수 설정
def convert_time_to_minutes(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes
df['Time'] = df['Time'].apply(convert_time_to_minutes)
# 날짜를 일수로 변환 - 22년 3월 2일로부터 일수 차이값으로 반환
# 뺄셈 연산을 위해 datetime 형식 변환
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Days_Since'] = (df['Date'] - pd.Timestamp("2022-03-02")) // pd.Timedelta('1D')
#불필요한 열 제거
df = df.drop(columns=['Date'])
#원-핫 인코딩
data = pd.get_dummies(df,columns=["Weather"])
# bool 값들을 0과 1로 변환
data[['Weather_1', 'Weather_2', 'Weather_3', 'Weather_4']] = data[['Weather_1', 'Weather_2', 'Weather_3', 'Weather_4']].astype(int)
print(data.head())

# 입력 변수와 출력 변수 분리
x = data.drop(columns=["Waiting_Passengers"])  # 입력 변수
y = data["Waiting_Passengers"]  # 출력 변수

# 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# XGBoost 회귀 모델 생성 (하이퍼파라미터 튜닝)
XG_model = XGBRegressor(
    n_estimators=600,  # 트리 개수
    objective='reg:squarederror',  # 목표 함수 MSE를 최소화하는 방향으로 모델을 학습
    learning_rate=0.01,  # 학습률
    max_depth=9,  # 트리 깊이
    min_child_weight=10,  # 최소 자식 노드 가중치
    gamma=5,  # 트리 노드를 분할할 때 요구되는 최소 손실 감소
    subsample=0.8,  # 각 트리를 학습할 때 사용할 데이터 샘플 비율 - 과적합 방지
    colsample_bytree=0.8,  # 피처 샘플링 비율
    reg_lambda=8  # L2 정규화 모델 복잡성을 제어하고 과적합을 방지
)
#모델 학습
XG_model.fit(x_train, y_train)

# 테스트 데이터에 대한 예측 수행
y_pred = XG_model.predict(x_test)

"""
#모델 평가
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mse = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'R²: {r2:.2f}')

# 6. 변수 중요도 확인 (Feature Importance)
importances = XG_model.feature_importances_
feature_names = x.columns

# 중요도 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='skyblue')
plt.title('Feature Importances in XGBOOST')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

"""
# 모델 저장
with open('xgboost_model.pkl', 'wb') as file:
    pickle.dump(XG_model, file)

