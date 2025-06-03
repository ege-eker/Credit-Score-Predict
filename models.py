from typing import cast, Literal
import joblib
import os
from sklearn.ensemble import  GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import torch
from nn_model import OrdinalNN
from xgboost import XGBClassifier
from pydantic import BaseModel
import numpy as np
import pandas as pd



def load_model_gb() -> GradientBoostingClassifier:
    model_path = "models/gb-model.pkl"
    if not os.path.exists(model_path):
        print("Model is not trained yet! Please run train.ipynb first.")
    return cast(GradientBoostingClassifier, joblib.load(model_path))

def load_model_rf() -> RandomForestClassifier:
    model_path = "models/rf-model.pkl"
    if not os.path.exists(model_path):
        print("Model is not trained yet! Please run train.ipynb first.")
    return cast(RandomForestClassifier, joblib.load(model_path))

def load_model_svm() -> SVC:
    model_path = "models/svm-model.pkl"
    if not os.path.exists(model_path):
        print("Model is not trained yet! Please run train.ipynb first.")
    return cast(SVC, joblib.load(model_path))

def load_model_nn() -> OrdinalNN:
    model_path = "models/nn-model.pth"
    if not os.path.exists(model_path):
        print("Model is not trained yet! Please run train.ipynb first.")
    model = OrdinalNN(17, 7)
    # whitelist OrdinalNN to allow loading the full model object safely
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_model_xgb() -> XGBClassifier:
    model_path = "models/xgb-model.ubj"
    if not os.path.exists(model_path):
        print("Model is not trained yet! Please run train.ipynb first.")
    xgb_model = XGBClassifier()
    xgb_model.load_model(model_path)
    return xgb_model

def load_scaler() -> StandardScaler:
    if not os.path.exists("models/scaler.pkl"):
        print("Model is not trained yet! Please run train.ipynb first.")
    return cast(StandardScaler, joblib.load("models/scaler.pkl"))


class CreditDetailsParams(BaseModel):
    income: int
    age: int
    employment_length: int
    loan_amount: int
    loan_intent: Literal["VENTURE", "PERSONAL", "MEDICAL", "HOMEIMPROVEMENT", "EDUCATION", "DEBTCONSOLIDATION"]
    home_ownership: Literal["OWN", "RENT", "MORTGAGE"]
    default_on_file: bool

    def to_dataframe(self, scaler: StandardScaler) -> pd.DataFrame:
        log_income = np.log1p(self.income)
        log_person_age = np.log1p(self.age)
        loan_percent_income = self.loan_amount / self.income if self.income != 0 else 0

        data = {
            "log_income": log_income,
            "log_person_age": log_person_age,
            "person_income": self.income,
            "person_age": self.age,
            "person_emp_length": self.employment_length,
            "loan_amnt": self.loan_amount,

            "loan_intent_VENTURE": float(self.loan_intent == "VENTURE"),
            "loan_intent_PERSONAL": float(self.loan_intent == "PERSONAL"),
            "loan_intent_MEDICAL": float(self.loan_intent == "MEDICAL"),
            "loan_intent_HOMEIMPROVEMENT": float(self.loan_intent == "HOMEIMPROVEMENT"),
            "loan_intent_EDUCATION": float(self.loan_intent == "EDUCATION"),
            "loan_intent_DEBTCONSOLIDATION": float(self.loan_intent == "DEBTCONSOLIDATION"),

            "person_home_ownership_OWN": float(self.home_ownership == "OWN"),
            "person_home_ownership_RENT": float(self.home_ownership == "RENT"),
            "person_home_ownership_MORTGAGE": float(self.home_ownership == "MORTGAGE"),

            "cb_person_default_on_file": float(self.default_on_file),

            "loan_percent_income": loan_percent_income
        }
        expected_order = [
            'log_income',
            'log_person_age',
            'person_income',
            'person_age',
            'person_emp_length',
            'loan_amnt',
            'loan_intent_VENTURE',
            'loan_intent_PERSONAL',
            'loan_intent_MEDICAL',
            'loan_intent_HOMEIMPROVEMENT',
            'loan_intent_EDUCATION',
            'loan_intent_DEBTCONSOLIDATION',
            'person_home_ownership_OWN',
            'person_home_ownership_RENT',
            'person_home_ownership_MORTGAGE',
            'cb_person_default_on_file',
            'loan_percent_income'
        ]
        df = pd.DataFrame([data])
        cont_cols = [
            "log_income",
            "log_person_age",
            "loan_percent_income",
            "person_income",
            "person_age",
            "person_emp_length",
            "loan_amnt",
        ]
        target_df = df[cont_cols]
        df_temp = df.drop(columns=cont_cols)
        target_df = pd.DataFrame(
            scaler.transform(target_df),
            columns=target_df.columns
        )

        df = pd.concat([
            df_temp,
            target_df
        ], axis=1)

        return df[expected_order]

    def to_tensor(self, scaler: StandardScaler) -> torch.Tensor:
        x = self.to_dataframe(scaler)
        return torch.tensor(x.values, dtype=torch.float32)

class MasterPredictor:
    def __init__(self):
        self.gb_model = load_model_gb()
        self.rf_model = load_model_rf()
        self.svm_model = load_model_svm()
        self.nn_model = load_model_nn()
        self.xgb_model = load_model_xgb()
        self.scaler = load_scaler()

    prediction_result = Literal["A", "B", "C", "D", "E", "F", "G"]

    def predict_gb(self, params: CreditDetailsParams) -> prediction_result:
        model = load_model_gb()
        x = params.to_dataframe(self.scaler)
        result = model.predict(x)[0]
        return self.map_raw_to_letter(result)

    def predict_rf(self, params: CreditDetailsParams) -> prediction_result:
        model = load_model_rf()
        x = params.to_dataframe(self.scaler)
        result = model.predict(x)[0]
        return self.map_raw_to_letter(result)

    def predict_svm(self, params: CreditDetailsParams) -> prediction_result:
        model = load_model_svm()
        x = params.to_dataframe(self.scaler)
        result = model.predict(x)[0]
        return self.map_raw_to_letter(result)

    def predict_nn(self, params: CreditDetailsParams) -> prediction_result:
        x = params.to_tensor(self.scaler)

        nn_model = load_model_nn()

        with torch.no_grad():
            _, predictions = nn_model(x)
            nn_ordinal_pred = predictions.cpu().numpy()

        result = nn_ordinal_pred[0]
        return self.map_raw_to_letter(result)

    def predict_xgb(self, params: CreditDetailsParams) -> prediction_result:
        model = load_model_xgb()
        x = params.to_dataframe(self.scaler)
        result = model.predict(x)[0]
        return self.map_raw_to_letter(result)

    @staticmethod
    def map_raw_to_letter(raw: int) -> prediction_result:
        if raw == 6: return "A"
        elif raw == 5: return "B"
        elif raw == 4: return "C"
        elif raw == 3: return "D"
        elif raw == 2: return "E"
        elif raw == 1: return "F"
        else: return "G"