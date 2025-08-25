FROM python:3.13-slim

ENV PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY src ./src

EXPOSE 8000

CMD ["uvicorn", "carni_detect.app:app", "--host", "0.0.0.0", "--port", "8000"]