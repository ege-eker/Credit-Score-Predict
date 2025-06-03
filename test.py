from models import CreditDetailsParams, load_scaler, load_model_nn, MasterPredictor

cred_param = CreditDetailsParams(
    age=25,
    income=9600,
    home_ownership="MORTGAGE",
    employment_length=5,
    loan_intent="EDUCATION",
    loan_amount=1000,
    default_on_file=False,
)

predictor = MasterPredictor()


result = predictor.predict_nn(cred_param)