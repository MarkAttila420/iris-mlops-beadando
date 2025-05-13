FROM python:3.12-slim

WORKDIR /app

# Install dependencies with proper error handling
RUN apt-get update && \
    apt-get install -y curl && \
    pip install --upgrade pip setuptools wheel && \
    pip install mlflow==2.22.0 psycopg2-binary

# Create required directories with appropriate permissions
RUN mkdir -p /app/mlruns /app/mlartifacts && \
    chmod -R 777 /app/mlruns /app/mlartifacts

# Expose port
EXPOSE 5000

# Use the JSON array format for CMD (recommended by Docker)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "/app/mlruns", "--default-artifact-root", "/app/mlartifacts"]