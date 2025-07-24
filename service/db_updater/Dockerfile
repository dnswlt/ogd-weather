FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy charts code (needed as a library)
COPY service/charts /app/service/charts

# Copy db-updater main script
COPY service/db_updater /app/service/db_updater

ENV OGD_BASE_DIR=/app/efs

# Run the new entrypoint from db-updater
CMD ["python3", "-m", "service.db_updater.app"]
