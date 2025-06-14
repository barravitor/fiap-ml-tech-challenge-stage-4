# api/Dockerfile

FROM python:3.10.12-slim

WORKDIR /app

COPY ../requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r requirements.txt

COPY ./api/app /app/app
COPY ./api/start_api.sh /app/start_api.sh
COPY ../shared /app/shared

EXPOSE 8000

CMD ["./start_api.sh"]