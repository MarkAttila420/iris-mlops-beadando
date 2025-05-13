from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
import mlflow
import mlflow.sklearn
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Iris Classifier API", 
              description="API for Iris flower classification",
              version="1.0.0")

# Environment variables with defaults
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "model.joblib"))
SCALER_PATH = os.getenv("SCALER_PATH", os.path.join(os.path.dirname(__file__), "..", "scaler.joblib"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Try to load model from MLflow first, fallback to local files
try:
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load the latest production model
    model = mlflow.sklearn.load_model("models:/IrisClassifier/Production")
    logger.info("Model loaded from MLflow")
    
    # For simplicity, we'll still use the local scaler
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"Scaler loaded from {SCALER_PATH}")
except Exception as e:
    logger.warning(f"Couldn't load model from MLflow: {str(e)}")
    logger.info("Falling back to local model files")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

# Pydantic model for request validation
class Features(BaseModel):
    features: list
    
    class Config:
        schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

# Pydantic model for response
class Prediction(BaseModel):
    prediction: int
    prediction_timestamp: str
    feature_names: list = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    model_version: str

@app.post("/predict", response_model=Prediction)
def predict(data: Features):
    """
    Make a prediction with the Iris classifier
    """
    try:
        # Validate input dimensions
        if len(data.features) != 4:
            raise HTTPException(status_code=400, detail="Input features must have exactly 4 values")
            
        # Transform input data
        features_scaled = scaler.transform([data.features])
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        # Log prediction
        logger.info(f"Prediction made: {prediction[0]} for features {data.features}")
        
        # Get model version (if from MLflow)
        try:
            model_version = mlflow.sklearn.get_model_info(model).version
        except:
            model_version = "unknown"
            
        # Return prediction with timestamp
        return Prediction(
            prediction=int(prediction[0]),
            prediction_timestamp=datetime.now().isoformat(),
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/")
def root():
    """API root with documentation link"""
    return {
        "message": "Welcome to the Iris Classifier API",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)