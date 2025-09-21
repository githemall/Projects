from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# FastAPI 앱 생성
app = FastAPI()

# -------------------- [모델 로딩] --------------------
# 서버가 시작될 때 모델을 한 번만 불러옵니다.
MODEL_NAME = "beomi/kcELECTRA-base-v2022"
MODEL_PATH = "/home/asd/projects/Portfolio/project2/Movie_review/best_model_state.bin" # 저장된 모델 파일 경로

# 토크나이저와 모델을 불러옵니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# 저장된 모델의 가중치(state_dict)를 불러와 적용합니다.
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
# CPU를 사용하도록 설정합니다. GPU에서 학습했더라도 CPU 환경에서 서빙할 수 있습니다.
device = torch.device("cpu")
model.to(device)
model.eval() # 모델을 평가 모드로 설정
# ----------------------------------------------------


# 입력 데이터 형식을 정의하는 Pydantic 모델
class SentimentRequest(BaseModel):
    text: str

# 예측 결과를 정의하는 Pydantic 모델
class SentimentResponse(BaseModel):
    prediction: str
    probability: float


@app.get("/")
def read_root():
    return {"message": "Movie Review Sentiment Analysis API"}


@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest):
    # 1. 입력된 텍스트를 받습니다.
    text = request.text
    
    # 2. 텍스트를 토큰화합니다.
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    )
    
    # 3. 모델 예측
    with torch.no_grad():
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs)
        
    # 4. 예측 결과를 확률로 변환하고 해석합니다.
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # 가장 확률이 높은 클래스를 찾습니다.
    prediction_idx = torch.argmax(probs).item()
    probability = probs[0, prediction_idx].item()
    
    prediction = "긍정 😄" if prediction_idx == 1 else "부정 😠"
    
    # 5. 결과를 반환합니다.
    return {
        "prediction": prediction,
        "probability": probability
    }