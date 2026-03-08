# Docker-based isolation for the OpenEnv MCP NegotiationEnvironment server.
# Build:  docker build -t negotiation-arena .
# Run:    docker run -p 8000:8000 negotiation-arena
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY server/ server/
COPY client/ client/
COPY reward.py .
COPY challenger.py .
COPY openenv.yaml .
COPY pyproject.toml .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
