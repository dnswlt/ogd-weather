from contextlib import asynccontextmanager
import datetime
import logging
import os
import sqlite3
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from zoneinfo import ZoneInfo
from . import charts
from . import db
from . import models
from . import logging_config as _  # configure logging


class ChartRequest(BaseModel):
    city: str
    chart_type: str


logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: open SQLite connection once
    base_dir = os.environ.get("OGD_BASE_DIR", ".")
    db_path = os.path.join(base_dir, db.DATABASE_FILENAME)
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


@app.get("/health")
def health():
    """Health check endpoint for cloud deployments."""
    return {"status": "ok"}


@app.get("/stations/{station_abbr}/charts/{chart_type}")
async def get_chart(
    station_abbr: str,
    chart_type: str,
    period: str = "6",
    from_year: str | None = None,
    to_year: str | None = None,
    window: str | None = None,
):
    station_abbr = station_abbr.upper()

    from_year_int = int(from_year) if from_year and from_year.isdigit() else None
    to_year_int = int(to_year) if to_year and to_year.isdigit() else None
    window_int = int(window) if window and window.isdigit() else None

    if chart_type == "temperature":
        df = db.read_daily_measurements(
            app.state.db,
            station_abbr,
            period=period,
            columns=[db.TEMP_DAILY_MIN, db.TEMP_DAILY_MEAN, db.TEMP_DAILY_MAX],
            from_year=from_year_int,
            to_year=to_year_int,
        )
        return {
            "vega_spec": charts.temperature_chart(
                df, station_abbr, period=period, window=window_int
            ),
        }
    elif chart_type == "temperature_deviation":
        df = db.read_daily_measurements(
            app.state.db,
            station_abbr,
            period=period,
            columns=[db.TEMP_DAILY_MEAN],
            from_year=from_year_int,
            to_year=to_year_int,
        )
        return {
            "vega_spec": charts.temperature_deviation_chart(
                df, station_abbr, period=period, window=window_int
            ),
        }
    elif chart_type == "precipitation":
        df = db.read_daily_measurements(
            app.state.db,
            station_abbr,
            period=period,
            columns=[db.PRECIP_DAILY_MM],
            from_year=from_year_int,
            to_year=to_year_int,
        )
        return {
            "vega_spec": charts.precipitation_chart(
                df, station_abbr, period=period, window=window_int
            ),
        }

    valid_charts = ["temperature", "precipitation", "temperature_deviation"]
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Invalid chart type (must be one of [{','.join(valid_charts)}])",
    )


@app.get("/stations/{station_abbr}/daily")
async def get_daily_measurements(
    station_abbr: str,
    date: str | None = None,
):
    station_abbr = station_abbr.upper()
    # Parse date, if specified, else assume yesterday.
    d = (
        datetime.date.fromisoformat(date)
        if date
        else datetime.date.today() - datetime.timedelta(days=1)
    )
    from_date = datetime.datetime(
        d.year, d.month, d.day, 1, 0, 0, 0, tzinfo=ZoneInfo("Europe/Zurich")
    )
    to_date = from_date + datetime.timedelta(days=1)
    df = db.read_hourly_measurements(
        app.state.db, station_abbr, from_date=from_date, to_date=to_date
    )
    return {
        "data": charts.daily_measurements(df, station_abbr),
    }


@app.get("/stations/{station_abbr}/summary")
async def get_summary(
    station_abbr: str,
    period: str = "6",
    from_year: str | None = None,
    to_year: str | None = None,
):
    station_abbr = station_abbr.upper()

    from_year_int = int(from_year) if from_year and from_year.isdigit() else None
    to_year_int = int(to_year) if to_year and to_year.isdigit() else None

    df = db.read_daily_measurements(
        app.state.db,
        station_abbr,
        period=period,
        columns=[
            db.TEMP_DAILY_MIN,
            db.TEMP_DAILY_MEAN,
            db.TEMP_DAILY_MAX,
            db.PRECIP_DAILY_MM,
        ],
        from_year=from_year_int,
        to_year=to_year_int,
    )
    stats = charts.station_stats(df, station_abbr, period=period)

    station = db.read_station(app.state.db, station_abbr)

    return {
        "summary": models.StationSummary(
            station=station,
            stats=stats,
        ),
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
