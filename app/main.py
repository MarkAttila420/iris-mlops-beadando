# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "scaler.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Új osztály, amit a FastAPI vár
class Features(BaseModel):
    features: list

@app.post("/predict")
def predict(data: Features):
    features_scaled = scaler.transform([data.features])
    prediction = model.predict(features_scaled)
    return {"prediction": int(prediction[0])}
