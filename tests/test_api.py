import pytest
import sys
import os
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Add the parent directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import app

client = TestClient(app)

def test_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid_input():
    """Test the prediction with valid input"""
    # Valid setosa features
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "prediction_timestamp" in data
    # Assuming setosa is class 0
    assert data["prediction"] == 0

def test_predict_invalid_input():
    """Test the prediction with invalid input"""
    # Too few features
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5]}
    )
    assert response.status_code == 400

def test_predict_boundary_case():
    """Test prediction with boundary case values"""
    # Edge case values
    response = client.post(
        "/predict",
        json={"features": [0.0, 0.0, 0.0, 0.0]}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()