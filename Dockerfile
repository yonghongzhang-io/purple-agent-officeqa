FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "a2a-sdk[http-server]>=0.3.20" \
    "httpx>=0.28.1" \
    "pydantic>=2.11.0" \
    "python-dotenv>=1.1.1" \
    "uvicorn>=0.38.0" \
    "anthropic>=0.40.0" \
    "openai>=1.50.0"

COPY src/ ./

EXPOSE 9009
CMD ["sh", "-c", "python server.py --host 0.0.0.0 --port ${PORT:-9009}"]
