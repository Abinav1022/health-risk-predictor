import joblib
import pandas as pd

heart_model = joblib.load("C:\\Users\\abina\\Desktop\\Health-risk-predictor\\models\\heart_model.pkl")
diabetes_model = joblib.load("C:\\Users\\abina\\Desktop\\Health-risk-predictor\\models\\diabetes_model.pkl")

def predict_heart(input_dict):
    return heart_model.predict(pd.DataFrame([input_dict]))[0]

def predict_diabetes(input_dict):
    return diabetes_model.predict(pd.DataFrame([input_dict]))[0]
