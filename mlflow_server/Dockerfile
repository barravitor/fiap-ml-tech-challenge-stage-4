FROM python:3.10-slim

# Installing system dependencies
RUN apt-get update && apt-get install -y nginx apache2-utils supervisor

# Copy the code and configuration files
WORKDIR /app
COPY . /app

RUN useradd -ms /bin/bash mlflowuser && \ chown -R mlflowuser:mlflowuser /app

USER mlflowuser

# Copy the nginx password and config file to where nginx expects it
COPY .htpasswd /etc/nginx/.htpasswd
COPY nginx.conf /etc/nginx/sites-enabled/default

# Install mlflow and other Python libs
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Expose port 80 (nginx)
EXPOSE 80

# Copy the supervisord.conf file to the expected location
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Run supervisor to manage nginx + mlflow
CMD ["supervisord", "-n"]
