import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_iris
from evidently.report import Report
from evidently.metrics import DataDriftTable, DataQualityTable, ClassificationPerformanceTable
from evidently.metrics.base_metric import LoadedMetric
from evidently.test_suite import TestSuite
from evidently.tests.base_test import LoadedTest
from evidently.test_preset import DataDriftTestPreset
from sklearn.model_selection import train_test_split

# Load the model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

# Load iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

# Create reference and current datasets
# For this example, we'll use a split of the same data
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
report.save_html("report.html")
print("Report generated successfully at 'report.html'")
