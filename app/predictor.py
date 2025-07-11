import pandas as pd
import joblib

# ✅ Use relative paths instead of full Windows paths
heart_model = joblib.load("models/heart_model.pkl")
diabetes_model = joblib.load("models/diabetes_model.pkl")

def predict_heart(input_dict):
    input_df = pd.DataFrame([input_dict])
    return heart_model.predict(input_df)[0]

def predict_diabetes(input_dict):
    input_df = pd.DataFrame([input_dict])
    return diabetes_model.predict(input_df)[0]
