import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load trained models
# -----------------------------
heart_model = joblib.load("C:\\Users\\abina\\Desktop\\Health-risk-predictor\\models\\heart_model.pkl")
diabetes_model = joblib.load("C:\\Users\\abina\\Desktop\\Health-risk-predictor\\models\\diabetes_model.pkl")

# -----------------------------
# Load background data
# -----------------------------
heart_data = pd.read_csv("C:\\Users\\abina\\Desktop\\Health-risk-predictor\\data\\heart.csv").drop("target", axis=1)
diabetes_data = pd.read_csv("C:\\Users\\abina\\Desktop\\Health-risk-predictor\\data\\diabetes.csv").drop("Outcome", axis=1)

# -----------------------------
# Initialize SHAP explainers
# -----------------------------
heart_explainer = shap.Explainer(heart_model.predict_proba, heart_data)
db_explainer = shap.Explainer(diabetes_model.predict_proba, diabetes_data)

# -----------------------------
# SHAP Explanation Functions
# -----------------------------

def explain_heart(input_df):
    """
    Generates and draws a SHAP waterfall plot for heart disease risk.
    Does not call st.pyplot() inside this function.
    """
    shap_values = heart_explainer(input_df)

    explanation = shap.Explanation(
        values=shap_values.values[0][1],                      # Class 1: Risk
        base_values=shap_values.base_values[0][1],
        data=input_df.iloc[0],
        feature_names=input_df.columns
    )

    shap.plots.waterfall(explanation, show=False)  # Draws into matplotlib
    # Streamlit will pick up this figure via plt.gcf()


def explain_diabetes(input_df):
    """
    Generates and draws a SHAP waterfall plot for diabetes risk.
    Does not call st.pyplot() inside this function.
    """
    shap_values = db_explainer(input_df)

    explanation = shap.Explanation(
        values=shap_values.values[0][1],                      
        base_values=shap_values.base_values[0][1],
        data=input_df.iloc[0],
        feature_names=input_df.columns
    )

    shap.plots.waterfall(explanation, show=False)
    
