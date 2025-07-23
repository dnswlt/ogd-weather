# --- builder stage: fetch data & build DB ---
FROM python:3.12-slim AS builder

WORKDIR /build

# Install deps required by ogd.py.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts needed to download CSV and build the sqlite DB.
COPY *.py ./service/charts/

# Run the CSV downloader and create /build/data/swissmetnet.sqlite.
RUN mkdir -p /build/data && \
    OGD_BASE_DIR=/build/data python3 -m service.charts.ogd /build/data

# --- runtime stage ---
FROM python:3.12-slim

WORKDIR /app

# Runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app into proper package directory.
RUN mkdir -p service/charts
COPY . ./service/charts

# Copy generated DB from builder.
COPY --from=builder /build/data/swissmetnet.sqlite /app/data/swissmetnet.sqlite

EXPOSE 8080
CMD ["uvicorn", "service.charts.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
