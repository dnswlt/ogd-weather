# ogd-weather

Fun with Open Data from MeteoSwiss.

## Services

### Initial setup

Prepare the Python virtual environment and install dependencies:

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Also make sure that all required environment variables are exported:

```bash
export OGD_BASE_DIR="$(pwd)/data"
```

### db-updater

If this is the first time you run the server, you'll need to download the
Open Government Data weather data:

```bash
source .venv/bin/activate
python3 -m service.db_updater.app
```

Weather CSV data gets downloaded to `$OGD_BASE_DIR` and an sqlite3 database `swissmetnet.sqlite`
gets created from that data.

You can run `python3 -m service.db_updater.app` repeatedly to update the DB with the latest data.

### charts

Run the server via `uvicorn`:

```bash
source .venv/bin/activate
# To use the service:
uvicorn service.charts.app:app --host 127.0.0.1 --port 8000 --workers 1
# During development, for fast reloads and more logging:
uvicorn service.charts.app:app --host 127.0.0.1 --port 8000 --workers 1 --log-level debug --reload
```

### api

The API server is a Go backend that sits in front of the Python charts server
and serves HTML pages.

```bash
cd service/api
go run ./cmd/server
```

The frontend lives in service/api/web/ and uses Vite for bundling.

#### First-time setup

```bash
cd service/api/web
npm install
```

#### Build once

After making changes, recreate the UI artifacts (`bundle.js`, `main.css`, etc.):

```bash
npm run build
```

#### Watch for changes (dev)

```bash
npm run dev
```

This builds static assets into service/api/static/dist/, which the Go server serves at /static/dist/.

## Docker

You can run both services in Docker using `docker compose`.  
A `docker-compose.yml` is provided for local integration. It builds both images and connects them on a shared network.

### Build images

```bash
make rebuild
```

### Run services

```bash
make up
```

This starts:  

- **Python charts service** on <http://localhost:8082>  
- **Go API service** on <http://localhost:8081>  

The Go backend is automatically configured to call the charts service via the internal Docker network.
