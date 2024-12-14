import pickle
from fastapi import FastAPI
import xgboost as xgb
from Datapreprocessor import DataPreprocessor
import pandas as pd
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 요청 받을 데이터의 구조 정의
class ChatRequest(BaseModel):
    user_id: str
    message: str

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서의 요청을 허용 (보안을 위해 필요한 도메인만 설정하는 것이 좋습니다)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, OPTIONS 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

# 전처리 객체 생성
preprocessor = DataPreprocessor()

# 저장된 XGBoost 모델 로드
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# 요청 데이터를 받기 위한 Pydantic 모델 생성
class PredictionInput(BaseModel):
    Date: str
    Day: int
    Time: str
    Weather: int
    Event: int
    Train_Arrival: int

@app.get("/")
async def read_root():
    return {"message": "Server is running!"}

@app.post("/predict")
async def predict(request: PredictionInput):
    # 요청 데이터 확인
    request_data = request.dict()
    
    # 요청 데이터를 데이터프레임으로 변환
    df = pd.DataFrame([request_data])
    print("데이터프레임 형성 후:")
    print(df)
    
    # 입력 데이터 전처리 수행
    df = preprocessor.preprocess(df)
    print("전처리된 데이터프레임:")
    print(df)
    
    # 예측 수행
    prediction = model.predict(df)
    print("예측 결과:", int(prediction[0]))
    
    # 예측 결과 반환
    return {"prediction": int(prediction[0])}

