# api/Dockerfile

FROM python:3.10.12-slim

WORKDIR /app

COPY ../requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY ./api/app /app/app

COPY ../shared /app/shared

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]