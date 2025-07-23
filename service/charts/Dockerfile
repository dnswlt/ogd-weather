FROM python:3.12-slim

WORKDIR /app

# Runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app into proper package directory.
RUN mkdir -p service/charts
COPY ./*.py ./service/charts

EXPOSE 8080
CMD ["uvicorn", "service.charts.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
