# ogd-weather

Fun with Open Data from MeteoSwiss.

## Services

### charts

Prepare the Python virtual environment and install dependencies:

```bash
pushd service/charts
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
popd
```

If this is the first time you run the server, you'll need to download the
Open Government Data weather data first:

```bash
export OGD_BASE_DIR=~/tmp/ogd-weather
python3 service/charts/ogd.py "$OGD_BASE_DIR"
```

Run the server via `uvicorn` (set `OGD_BASE_DIR` to the directory you downloaded the CSV files into):

```bash
source service/charts/.venv/bin/activate
OGD_BASE_DIR=~/tmp/ogd-weather uvicorn service.charts.app:app --host 127.0.0.1 --port 8000 --workers 1
```

### api

```bash
cd service/api
go run ./cmd/server
```

## Docker

You can run both services in Docker, either individually or together with `docker compose`.  

### Build images

From the repo root:  

```bash
# Build Python charts service
docker build -t weather-charts service/charts

# Build Go API service
docker build -t weather-api service/api
```

### Run services with docker compose

A `docker-compose.yml` is provided for local integration. It builds both images and connects them on a shared network.

```bash
docker compose up --build
```

This starts:  

- **Python charts service** on <http://localhost:8082>  
- **Go API service** on <http://localhost:8081>  

The Go backend is automatically configured to call the charts service via the internal Docker network.

If you change application code:  

- `docker compose up --build` – rebuilds only affected layers  
- `docker compose up --build --force-recreate` – full rebuild (needed after Dockerfile or dependency changes)

This setup matches production packaging:  

- Python container includes the prebuilt SQLite database (`swissmetnet.sqlite`)  
- Go container includes HTML templates  
