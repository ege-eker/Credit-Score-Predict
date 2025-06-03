from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from models import CreditDetailsParams, MasterPredictor

app = FastAPI()


class PredictionRequest(BaseModel):
    params: CreditDetailsParams
    model: Literal["gb", "lgb", "nn", "rf", "svm", "xgb"]

predictor = MasterPredictor()

@app.post("/predict")
def predict(body: PredictionRequest):
    if body.model == "gb": return predictor.predict_gb(body.params)
    elif body.model == "lgb": return predictor.predict_lgb(body.params)
    elif body.model == "nn": return predictor.predict_nn(body.params)
    elif body.model == "rf": return predictor.predict_rf(body.params)
    elif body.model == "svm": return predictor.predict_svm(body.params)
    else: return predictor.predict_xgb(body.params)

uvicorn.run(app, port=3000, host="0.0.0.0")