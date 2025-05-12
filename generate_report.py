from evidently.metrics import ColumnDriftMetric
from evidently.report import Report
from sklearn.datasets import load_iris
import pandas as pd

# Load Iris data
iris = load_iris(as_frame=True)
X = iris.frame.drop(columns=[iris.target.name])  # just features

# Build a Report for all columns
report = Report(metrics=[
    ColumnDriftMetric(column_name=col) for col in X.columns
])

# Run reference vs. current (here they’re identical)
report.run(reference_data=X, current_data=X)

# Save HTML
report.save_html("report.html")
