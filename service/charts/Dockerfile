FROM python:3.12-slim

# Install curl (for health checks) and CA certs, then clean up apt cache
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code
COPY service/charts /app/service/charts

EXPOSE 8080

CMD ["uvicorn", "service.charts.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
