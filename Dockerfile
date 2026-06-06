FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy everything needed for install first
COPY pyproject.toml ./
COPY README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

# Copy rest of source
COPY . .

EXPOSE 8000

CMD ["rag-graph-api", "--host", "0.0.0.0", "--port", "8000"]