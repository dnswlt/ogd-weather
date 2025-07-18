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
