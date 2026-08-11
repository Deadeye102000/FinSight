FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FINSIGHT_API_URL=http://localhost:8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .
COPY docker/supervisord.conf /etc/supervisor/conf.d/finsight.conf

EXPOSE 8000 8501

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/finsight.conf"]
