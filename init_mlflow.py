import mlflow
import sys
import time

def initialize_mlflow(max_retries=10, retry_delay=2):
    """
    Initialize MLflow with experiments and configurations
    """
    print("Initializing MLflow...")
    mlflow_uri = "http://localhost:5000"
    
    # Try to connect to MLflow server
    for attempt in range(max_retries):
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.create_experiment("Iris_Classification")
            print(f"Successfully created experiment 'Iris_Classification'")
            return True
        except Exception as e:
            print(f"Attempt {attempt+1}/{max_retries}: Cannot connect to MLflow: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    print("Failed to initialize MLflow after maximum retries")
    return False

if __name__ == "__main__":
    if initialize_mlflow():
        sys.exit(0)
    else:
        sys.exit(1)