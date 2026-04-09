FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first — cached layer, only busts when requirements.txt changes
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source last — changes here don't re-trigger dep install
COPY . .
RUN pip install . --no-deps

EXPOSE 8000
