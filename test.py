from models import CreditDetailsParams, load_scaler, load_model_nn, MasterPredictor

input = CreditDetailsParams(
    age=25,
    income=9600,
    home_ownership="RENT",
    employment_length=5,
    loan_intent="N",
    loan_amount=1000,
    default_on_file=False,
)

predictor = MasterPredictor()


print(predictor.predict_nn(input))