# api/Dockerfile

FROM python:3.10.12-slim

WORKDIR /app

COPY ../requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && pip install --no-cache-dir -r requirements.txt

COPY ./api/app /app/app

COPY ../shared /app/shared

RUN echo "$GOOGLE_CREDENTIALS_JSON" > ./gcp-storage-service-account.json && ls -l "$GOOGLE_APPLICATION_CREDENTIALS"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]