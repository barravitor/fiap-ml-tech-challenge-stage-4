# mlflow_server/Dockerfile

FROM python:3.10.12-slim

USER root

# Installing system dependencies
RUN apt-get update && apt-get install -y nginx apache2-utils supervisor wget

RUN rm -rf /var/lib/apt/lists/*

# Copy the code and configuration files
WORKDIR /app
COPY . /app

# Install mlflow and other Python libs
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt






# Criar diretórios centralizados em /app/data
RUN mkdir -p /app/data/minio-data /app/data/mlartifacts /app/data/logs /app

# Baixar o servidor MinIO
RUN wget https://dl.min.io/server/minio/release/linux-amd64/minio && \
    chmod +x minio && mv minio /usr/local/bin/

# Variáveis de ambiente para o MLflow usar o MinIO
ENV MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
ENV AWS_ACCESS_KEY_ID=minioadmin
ENV AWS_SECRET_ACCESS_KEY=minioadmin
ENV MINIO_ROOT_USER=minioadmin
ENV MINIO_ROOT_PASSWORD=minioadmin






# Copy the nginx password and config file to where nginx expects it
COPY .htpasswd /etc/nginx/.htpasswd
COPY nginx.conf /etc/nginx/sites-enabled/default

# Copy the supervisord.conf file to the expected location
COPY supervisord.conf /etc/supervisor/supervisord.conf

# Expose port 80 (nginx)
EXPOSE 80

# Run supervisor to manage nginx + mlflow
CMD ["supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
