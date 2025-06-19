from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from models import CreditDetailsParams, MasterPredictor

app = FastAPI()


class PredictionRequest(BaseModel):
    params: CreditDetailsParams
    model: Literal["gb", "lgb", "nn", "rf", "svm", "xgb"]

class PredictionResponse(BaseModel):
    result: str

predictor = MasterPredictor()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://credit.umceko.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/predict")
def predict(body: PredictionRequest):
    if body.model == "gb":
        return PredictionResponse(result=predictor.predict_gb(body.params))
    elif body.model == "lgb":
        return PredictionResponse(result=predictor.predict_lgb(body.params))
    elif body.model == "nn":
        return PredictionResponse(result=predictor.predict_nn(body.params))
    elif body.model == "rf":
        return PredictionResponse(result=predictor.predict_rf(body.params))
    elif body.model == "svm":
        return PredictionResponse(result=predictor.predict_svm(body.params))
    else:
        return PredictionResponse(result=predictor.predict_xgb(body.params))


uvicorn.run(app, port=3000, host="0.0.0.0")
