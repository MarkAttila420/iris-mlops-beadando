# airflow/dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    joblib.dump(model, "/tmp/model.joblib")
    joblib.dump(scaler, "/tmp/scaler.joblib")

with DAG("iris_training_dag", start_date=datetime(2023, 1, 1), schedule="@daily", catchup=False) as dag:
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )
