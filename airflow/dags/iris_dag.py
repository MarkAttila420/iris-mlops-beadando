from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import mlflow

# Base directory for where to save files
BASE_DIR = "/opt/airflow/dags/outputs"
os.makedirs(BASE_DIR, exist_ok=True)

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")

def train_model():
    """Train a new Iris classifier model and save it"""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://host.docker.internal:5000")
    mlflow.set_experiment("Iris_Classification")
    
    # Load and prepare data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model", registered_model_name="IrisClassifier")
    
    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Return paths for next task
    return {"model_path": MODEL_PATH, "scaler_path": SCALER_PATH}

def generate_report(**context):
    """Generate an Evidently report for the model"""
    # Get model paths from previous task
    task_instance = context['ti']
    paths = task_instance.xcom_pull(task_ids='train_model_task')
    
    model_path = paths['model_path']
    scaler_path = paths['scaler_path']
    
    # Import here to avoid loading these libraries in the entire DAG
    import pandas as pd
    import numpy as np
    from evidently.report import Report
    from evidently.metrics import DataDriftTable, DataQualityTable, ClassificationPerformanceTable
    
    # Load the model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    
    # Create reference and current datasets
    X_ref, X_curr, y_ref, y_curr = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scale the data
    X_ref_scaled = pd.DataFrame(scaler.transform(X_ref), columns=X_ref.columns)
    X_curr_scaled = pd.DataFrame(scaler.transform(X_curr), columns=X_curr.columns)
    
    # Add predictions
    X_ref_scaled['prediction'] = model.predict(X_ref_scaled)
    X_curr_scaled['prediction'] = model.predict(X_curr_scaled)
    
    # Add target
    X_ref_scaled['target'] = y_ref.values
    X_curr_scaled['target'] = y_curr.values
    
    # Create an Evidently report
    report = Report(metrics=[
        DataDriftTable(),
        DataQualityTable(),
        ClassificationPerformanceTable(y_pred_name="prediction", y_true_name="target")
    ])
    
    # Calculate metrics
    report.run(reference_data=X_ref_scaled, current_data=X_curr_scaled)
    
    # Save the report
    report_path = os.path.join(BASE_DIR, "report.html")
    report.save_html(report_path)
    
    return {"report_path": report_path}

def copy_files_to_app_directory(**context):
    """Copy model, scaler, and report to the app directory"""
    # Get paths from previous tasks
    task_instance = context['ti']
    model_paths = task_instance.xcom_pull(task_ids='train_model_task')
    report_paths = task_instance.xcom_pull(task_ids='generate_report_task')
    
    # Use host path variables here
    host_app_dir = "/opt/airflow/dags/app_dir"  # This directory should be mounted in docker-compose
    os.makedirs(host_app_dir, exist_ok=True)
    
    # Copy files
    import shutil
    shutil.copy(model_paths['model_path'], os.path.join(host_app_dir, "model.joblib"))
    shutil.copy(model_paths['scaler_path'], os.path.join(host_app_dir, "scaler.joblib"))
    shutil.copy(report_paths['report_path'], os.path.join(host_app_dir, "report.html"))
    
    print(f"Files copied to {host_app_dir}")

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create the DAG
with DAG(
    dag_id="iris_mlops_pipeline",
    default_args=default_args,
    description='A pipeline for Iris model training and monitoring',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['mlops', 'iris'],
) as dag:
    
    # Task to train the model
    train_model_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model,
    )
    
    # Task to generate the report
    generate_report_task = PythonOperator(
        task_id='generate_report_task',
        python_callable=generate_report,
        provide_context=True,
    )
    
    # Task to copy files to app directory
    copy_files_task = PythonOperator(
        task_id='copy_files_task',
        python_callable=copy_files_to_app_directory,
        provide_context=True,
    )
    
    # Define task dependencies
    train_model_task >> generate_report_task >> copy_files_task