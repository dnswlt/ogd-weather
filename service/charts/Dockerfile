# --- builder stage: fetch data & build DB ---
FROM python:3.12-slim AS builder

WORKDIR /build

# Install deps required by ogd.py.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only ogd.py, needed to download CSV and build the sqlite DB.
COPY ogd.py .

# Run the CSV downloader and create /build/data/swissmetnet.sqlite.
RUN mkdir -p /build/data && \
    OGD_BASE_DIR=/build/data python ogd.py /build/data

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
ENV OGD_BASE_DIR=/app/data
CMD ["uvicorn", "service.charts.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
