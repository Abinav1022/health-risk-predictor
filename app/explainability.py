import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ✅ Load models using relative paths
heart_model = joblib.load("models/heart_model.pkl")
diabetes_model = joblib.load("models/diabetes_model.pkl")

# ✅ Load background data using relative paths
heart_data = pd.read_csv("data/heart.csv").drop("target", axis=1)
diabetes_data = pd.read_csv("data/diabetes.csv").drop("Outcome", axis=1)

# Initialize SHAP explainers
heart_explainer = shap.Explainer(heart_model.predict_proba, heart_data)
db_explainer = shap.Explainer(diabetes_model.predict_proba, diabetes_data)

# Heart disease explanation
def explain_heart(input_df):
    shap_values = heart_explainer(input_df)
    explanation = shap.Explanation(
        values=shap_values.values[0][1],
        base_values=shap_values.base_values[0][1],
        data=input_df.iloc[0],
        feature_names=input_df.columns
    )
    shap.plots.waterfall(explanation, show=False)  # Don't show yet — app.py will use plt.gcf()

# Diabetes explanation
def explain_diabetes(input_df):
    shap_values = db_explainer(input_df)
    explanation = shap.Explanation(
        values=shap_values.values[0][1],
        base_values=shap_values.base_values[0][1],
        data=input_df.iloc[0],
        feature_names=input_df.columns
    )
    shap.plots.waterfall(explanation, show=False)
