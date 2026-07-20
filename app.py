import streamlit as st
import pandas as pd
import joblib

# Load model, scaler and columns
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Heart Disease Prediction")

st.title("❤️ Heart Disease Prediction System")

age = st.number_input("Age", 1, 100)
sex = st.selectbox("Sex", ["M", "F"])
chest = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
bp = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
hr = st.number_input("Maximum Heart Rate")
angina = st.selectbox("Exercise Induced Angina", ["Y", "N"])
oldpeak = st.number_input("Old Peak")
slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):

    sample = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest],
        "RestingBP": [bp],
        "Cholesterol": [chol],
        "FastingBS": [fbs],
        "RestingECG": [ecg],
        "MaxHR": [hr],
        "ExerciseAngina": [angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [slope]
    })

    sample = pd.get_dummies(sample)

    sample = sample.reindex(columns=columns, fill_value=0)

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("❤️ Heart Disease : YES")
    else:
        st.success("💚 Heart Disease : NO")