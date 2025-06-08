# api/Dockerfile

FROM python:3.10.12-slim

WORKDIR /app

COPY ./api/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY ./api/app /app/app
COPY ./api/start_api.sh /app/start_api.sh
COPY ../shared /app/shared

EXPOSE 8000

CMD ["./start_api.sh"]