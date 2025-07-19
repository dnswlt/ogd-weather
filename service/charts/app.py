from contextlib import asynccontextmanager
import logging
import os
import sqlite3
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from . import charts
from . import db


class ChartRequest(BaseModel):
    city: str
    chart_type: str


# Global logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: open SQLite connection once
    base_dir = os.environ.get("OGD_BASE_DIR", ".")
    db_path = os.path.join(base_dir, "swissmetnet.sqlite")
    conn = sqlite3.connect(db_path, check_same_thread=True)
    logger.info("Connected to sqlite3 at %s", db_path)
    conn.row_factory = sqlite3.Row
    app.state.db = conn

    # If charts module needs to initialize something
    # charts.init_db(conn)

    yield

    # Shutdown: close DB connection cleanly
    conn.close()


# Always create the app, we're running this thing with uvicorn ONLY.
app = FastAPI(lifespan=lifespan)


@app.exception_handler(charts.StationNotFoundError)
async def station_not_found_handler(request, exc: charts.StationNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


@app.exception_handler(charts.NoDataError)
async def no_data_error_handler(request, exc: charts.NoDataError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


@app.get("/stations/{station_abbr}/charts/{chart_type}")
async def get_chart(station_abbr: str, chart_type: str, month: int = 6):
    station_abbr = station_abbr.upper()
    if chart_type == "temperature":
        df = db.read_daily_historical(
            app.state.db,
            station_abbr,
            month=month,
            columns=[db.TEMP_DAILY_MIN, db.TEMP_DAILY_MEAN, db.TEMP_DAILY_MAX],
        )
        return charts.temperature_chart(df, station_abbr, month=month)
    elif chart_type == "precipitation":
        df = db.read_daily_historical(
            app.state.db, station_abbr, month=month, columns=[db.PRECIP_DAILY_MM]
        )
        return charts.precipitation_chart(df, station_abbr, month=month)

    valid_charts = ["temperature", "precipitation"]
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid chart type (must be one of [{','.join(valid_charts)}])",
    )


@app.get("/stations/{station_abbr}/stats")
async def get_stats(
    station_abbr: str,
    month: int = 6,
    from_year: int | None = None,
    to_year: int | None = None,
):
    station_abbr = station_abbr.upper()
    df = db.read_daily_historical(
        app.state.db,
        station_abbr,
        month=month,
        columns=[
            db.TEMP_DAILY_MIN,
            db.TEMP_DAILY_MEAN,
            db.TEMP_DAILY_MAX,
            db.PRECIP_DAILY_MM,
        ],
        from_year=from_year,
        to_year=to_year,
    )
    return {
        "stats": charts.weather_stats(df, station_abbr, month=month),
    }


@app.get("/stations")
async def list_stations(cantons: str | None = None):
    cantons_list = None
    if cantons:
        cantons_list = cantons.split(",")

    stations = db.read_stations(app.state.db, cantons=cantons_list, exclude_empty=True)
    return {
        "stations": stations,
    }
