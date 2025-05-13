import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
import json
import matplotlib.pyplot as plt
import time
import mlflow
from urllib.parse import urlparse

def check_mlflow_connection(max_retries=3, retry_delay=2):
    """Check MLflow connection with retries and display status in the dashboard"""
    mlflow_url = os.getenv("MLFLOW_URL", "http://mlflow:5000")
    st.sidebar.info(f"Attempting to connect to MLflow at: {mlflow_url}")
    
    # Try basic URL connection first
    connected = False
    base_response = None
    for attempt in range(max_retries):
        try:
            base_response = requests.get(f"{mlflow_url}/", timeout=5)
            base_status = base_response.status_code
            
            if base_status == 200:
                connected = True
                break
            else:
                st.sidebar.warning(f"Attempt {attempt+1}/{max_retries}: MLflow returned status {base_status}")
                
        except requests.RequestException as e:
            st.sidebar.warning(f"Attempt {attempt+1}/{max_retries}: {str(e)}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    if not connected:
        st.sidebar.error("❌ Failed to connect to MLflow server")
        if base_response:
            st.sidebar.error(f"Response code: {base_response.status_code}")
            st.sidebar.error(f"Response text: {base_response.text[:100]}...")
        return False
    
    # If base URL works, try API endpoint for experiments
    try:
        # Set the tracking URI for MLflow
        mlflow.set_tracking_uri(mlflow_url)
        # Get experiments to verify API works
        experiments = mlflow.search_experiments()
        
        st.sidebar.success(f"✅ Successfully connected to MLflow at {mlflow_url}")
        st.sidebar.success(f"Found {len(experiments)} experiments")
        
        # Show experiment names
        if experiments:
            experiment_names = [exp.name for exp in experiments]
            st.sidebar.write("Available experiments:")
            for name in experiment_names:
                st.sidebar.write(f"- {name}")
        
        return True
        
    except Exception as e:
        st.sidebar.warning("⚠️ MLflow base URL works but API returned error")
        st.sidebar.warning(str(e))
        return False

# Set page config
st.set_page_config(
    page_title="Iris Prediction Dashboard",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Prediction Dashboard")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Predictions", "Model Monitoring", "About"])

if st.sidebar.button("Test MLflow Connection"):
    check_mlflow_connection()

# API Endpoint
API_URL = os.getenv("API_URL", "http://api:8000")

if page == "Predictions":
    st.header("Make Predictions")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Enter Iris Features")
        sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
        sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
        petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
        petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)
        
        # Create feature array
        features = [sepal_length, sepal_width, petal_length, petal_width]
        
        # Map prediction to class name
        class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
        
        if st.button("🔮 Predict"):
            try:
                # Make prediction via API
                response = requests.post(
                    f"{API_URL}/predict", 
                    json={"features": features}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    prediction_class = class_names.get(result['prediction'], f"Unknown ({result['prediction']})")
                    
                    # Show prediction with success banner
                    st.success(f"Prediction: **{prediction_class.capitalize()}**")
                    st.info(f"Model version: {result.get('model_version', 'unknown')}")
                    st.info(f"Prediction time: {result.get('prediction_timestamp', 'unknown')}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
    
    with col2:
        st.subheader("Iris Species")
        # Display images of iris species
        st.image("https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/images/iris.png", 
                 caption="Iris Species: Setosa, Versicolor, and Virginica")
        
        # Show feature explanation
        st.subheader("Feature Information")
        feature_df = pd.DataFrame({
            "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
            "Description": [
                "Length of the sepal (in cm)",
                "Width of the sepal (in cm)",
                "Length of the petal (in cm)",
                "Width of the petal (in cm)"
            ],
            "Typical Range": ["4.3-7.9", "2.0-4.4", "1.0-6.9", "0.1-2.5"]
        })
        st.table(feature_df)

elif page == "Model Monitoring":
    st.header("Model Monitoring Dashboard")
    
    # Show model performance metrics
    st.subheader("Model Performance")
    
    # EvidentlyAI report embedding
    report_path = os.path.join(os.path.dirname(__file__), "..", "report.html")
    
    # Check if report exists
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_html = f.read()
        # Embed in iframe
        st.components.v1.html(report_html, height=1000, scrolling=True)
    else:
        st.error("Model monitoring report not found. Please generate the report first.")
        
        # Add a button to generate report
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                try:
                    # Use the standalone script to generate report
                    import subprocess
                    result = subprocess.run(
                        ["python", "../generate_report.py"], 
                        capture_output=True, 
                        text=True
                    )
                    if result.returncode == 0:
                        st.success("Report generated successfully! Please refresh the page.")
                    else:
                        st.error(f"Error generating report: {result.stderr}")
                except Exception as e:
                    st.error(f"Failed to generate report: {str(e)}")

else:  # About page
    st.header("About This Project")
    
    st.markdown("""
    ### MLOps Project for Iris Classification
    
    This project demonstrates a complete MLOps pipeline for a machine learning model 
    that classifies iris flowers based on their measurements.
    
    #### Components:
    
    - **FastAPI Backend**: Serving predictions through a REST API
    - **Streamlit Dashboard**: User interface for making predictions and viewing monitoring reports
    - **MLflow**: For model tracking and registry
    - **Evidently AI**: For model monitoring and reporting
    - **Airflow**: For workflow automation and scheduling
    
    #### Dataset:
    
    The famous Iris dataset contains measurements for three iris species:
    - Setosa
    - Versicolor
    - Virginica
    
    #### Model:
    
    A Random Forest classifier trained on the iris dataset.
    """)
    
    # Health check for services
    st.subheader("System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            response = requests.get(f"{API_URL}/health", timeout=2)
            if response.status_code == 200:
                st.success("✅ API is online")
            else:
                st.error("❌ API is not working properly")
        except:
            st.error("❌ API is offline")
    
    with col2:
        mlflow_url = os.getenv("MLFLOW_URL", "http://mlflow:5000")
        st.info(f"Attempting to connect to MLflow at: {mlflow_url}")
        try:
            # Try first with api/2.0 endpoint
            response = requests.get(f"{mlflow_url}/api/2.0/mlflow/experiments/list", timeout=5)
            if response.status_code == 200:
                st.success(f"✅ MLflow API is online")
            else:
                # Try just the root URL
                base_response = requests.get(mlflow_url, timeout=5)
                if base_response.status_code == 200:
                    st.warning(f"⚠️ MLflow base URL works but API returned: {response.status_code}")
                else:
                    st.error(f"❌ MLflow returned status code: {response.status_code}")
                st.code(response.text[:200] + "..." if len(response.text) > 200 else response.text)
        except Exception as e:
            st.error(f"❌ MLflow is offline: {str(e)}")
            st.info(f"Attempted to connect to: {mlflow_url}")
    
    with col3:
        # Check for report file existence
        report_path = os.path.join(os.path.dirname(__file__), "..", "report.html")
        if os.path.exists(report_path):
            st.success("✅ Monitoring report available")
        else:
            st.warning("⚠️ Monitoring report not generated yet")