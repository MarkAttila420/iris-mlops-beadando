import mlflow
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Connect to your MLflow server
mlflow_url = os.getenv("MLFLOW_URL", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_url)

# Create a new experiment
experiment_name = "Iris_Classification_V3"  # Change this name for different experiments
try:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Created new experiment '{experiment_name}' with ID: {experiment_id}")
except mlflow.exceptions.MlflowException as e:
    if "already exists" in str(e):
        print(f"Experiment '{experiment_name}' already exists")
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    else:
        raise e

# Load and prepare data
print("Loading dataset...")
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model parameters
n_estimators = 100
max_depth = 10
min_samples_split = 2

print(f"Training RandomForest model with {n_estimators} estimators...")

# Start a run in this experiment
with mlflow.start_run(experiment_id=experiment_id):
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Feature importance
    feature_importance = model.feature_importances_
    
    # Log parameters
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Log feature importance
    for i, importance in enumerate(feature_importance):
        # Clean feature name for MLflow compatibility
        clean_feature_name = feature_names[i].replace("(", "").replace(")", "").replace(" ", "_")
        mlflow.log_metric(f"feature_importance_{clean_feature_name}", importance)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # Log the scaler as an artifact
    import joblib
    scaler_path = "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(scaler_path)
    
    print(f"Successfully trained model with accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print("Model and metrics logged to MLflow")