import streamlit as st
import pandas as pd
from predictor import predict_heart, predict_diabetes
from explainability import explain_heart, explain_diabetes
import matplotlib.pyplot as plt

st.title("🩺 Healthcare Risk Predictor")
mode = st.selectbox("Select Disease Type", ["Heart Disease", "Diabetes"])

if mode == "Heart Disease":
    age = st.slider("Age", 20, 80)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.slider("Chest Pain Type", 0, 3)
    trestbps = st.slider("Resting BP", 80, 200)
    chol = st.slider("Cholesterol", 100, 400)
    fbs = st.selectbox("Fasting Sugar > 120", [0, 1])
    restecg = st.slider("Rest ECG", 0, 2)
    thalach = st.slider("Max HR", 60, 202)
    exang = st.selectbox("Exercise Angina", [0, 1])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0)
    slope = st.slider("Slope", 0, 2)
    ca = st.slider("Major vessels", 0, 4)
    thal = st.slider("Thal", 0, 3)

    inputs = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    if st.button("Predict Heart Risk"):
        result = predict_heart(inputs)
        st.success(f"Heart Disease Risk: {'Yes' if result == 1 else 'No'}")
        explain_heart(pd.DataFrame([inputs]))  # This should now return a fig
        # Use matplotlib safely
        fig = plt.gcf()
        st.pyplot(fig)

else:
    pregnancies = st.slider("Pregnancies", 0, 20)
    glucose = st.slider("Glucose", 0, 200)
    bp = st.slider("Blood Pressure", 0, 140)
    skin = st.slider("Skin Thickness", 0, 100)
    insulin = st.slider("Insulin", 0, 900)
    bmi = st.slider("BMI", 0.0, 70.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0)
    age = st.slider("Age", 20, 80)

    inputs = {
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
        "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age
    }

    if st.button("Predict Diabetes Risk"):
        result = predict_diabetes(inputs)
        st.success(f"Diabetes Risk: {'Yes' if result == 1 else 'No'}")
        explain_diabetes(pd.DataFrame([inputs]))  # This should now return a fig
        # Use matplotlib safely
        fig = plt.gcf()
        st.pyplot(fig)

    # Optional Custom Visualization
    if st.checkbox("Show Glucose vs Age Plot"):
        fig, ax = plt.subplots()
        ax.scatter([inputs['Age']], [inputs['Glucose']], color='red', label='You')
        ax.set_xlabel("Age")
        ax.set_ylabel("Glucose Level")
        ax.set_title("Your Glucose vs Age")
        ax.legend()
        st.pyplot(fig)
