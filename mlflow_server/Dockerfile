# mlflow_server/Dockerfile

FROM python:3.10.12-slim

USER root

# Installing system dependencies
RUN apt-get update && apt-get install -y nginx apache2-utils supervisor && rm -rf /var/lib/apt/lists/*

# Create directories to store MLflow data and artifacts
RUN mkdir -p /mlflow/mlruns && chmod -R 777 /mlflow

# Copy the code and configuration files
WORKDIR /app
COPY . /app

# Install mlflow and other Python libs
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the nginx password and config file to where nginx expects it
COPY .htpasswd /etc/nginx/.htpasswd
COPY nginx.conf /etc/nginx/sites-enabled/default

# Copy the supervisord.conf file to the expected location
COPY supervisord.conf /etc/supervisor/supervisord.conf

# Run supervisor to manage nginx + mlflow
CMD ["supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]

# Expose port 80 (nginx)
EXPOSE 80