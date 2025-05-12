import streamlit as st
import pandas as pd
import joblib
import os

st.title("Iris Prediction Dashboard")

# Modell és scaler betöltése
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "scaler.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Felhasználótól bemenet bekérése
st.write("## Upload your Iris features")
sepal_length = st.number_input("Sepal Length", 0.0, 10.0, 5.1)
sepal_width = st.number_input("Sepal Width", 0.0, 10.0, 3.5)
petal_length = st.number_input("Petal Length", 0.0, 10.0, 1.4)
petal_width = st.number_input("Petal Width", 0.0, 10.0, 0.2)

# Predikció gomb
if st.button("Predict"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    st.success(f"Prediction: {prediction[0]}")

# EvidentlyAI riport megjelenítése iframe-ben
st.write("## Model Monitoring Report")

report_path = os.path.join(os.path.dirname(__file__), "..", "report.html")

# Ellenőrizzük, hogy a riport létezik-e
if os.path.exists(report_path):
    with open(report_path, "r", encoding="utf-8") as f:
        report_html = f.read()
    # Iframe-be ágyazás
    st.components.v1.html(report_html, height=1000, scrolling=True)
else:
    st.error("Report not found. Please generate the report first.")
