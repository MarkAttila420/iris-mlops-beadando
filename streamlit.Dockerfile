FROM python:3.12-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY streamlit_app/ ./streamlit_app/
COPY model.joblib scaler.joblib ./

# Run the Streamlit application
CMD ["streamlit", "run", "streamlit_app/dashboard.py"]