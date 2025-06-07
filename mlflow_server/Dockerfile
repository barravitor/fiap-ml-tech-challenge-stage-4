# mlflow_server/Dockerfile

FROM python:3.10.12-slim

USER root

# Installing system dependencies
RUN apt-get update && apt-get install -y nginx apache2-utils supervisor

# Create directories to store MLflow data and artifacts
RUN mkdir -p /mlflow/mlruns && chmod 777 /mlflow/mlruns

# Copy the code and configuration files
WORKDIR /
COPY . /

# Copy the nginx password and config file to where nginx expects it
COPY .htpasswd /etc/nginx/.htpasswd
COPY nginx.conf /etc/nginx/sites-enabled/default

# Install mlflow and other Python libs
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose port 80 (nginx)
EXPOSE 80

# Copy the supervisord.conf file to the expected location
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Environment variables for MLflow
ENV MLFLOW_TRACKING_URI=file:///mlflow/mlruns
ENV MLFLOW_ARTIFACT_ROOT=file:///mlflow/mlruns

# Run supervisor to manage nginx + mlflow
CMD ["supervisord", "-n"]
