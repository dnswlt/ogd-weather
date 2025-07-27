from contextlib import asynccontextmanager
import datetime
import logging
import os
import sqlite3
from fastapi import FastAPI, HTTPException, status, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from zoneinfo import ZoneInfo
from . import charts
from . import db
from . import models
from . import logging_config as _  # configure logging
from .errors import NoDataError, StationNotFoundError


class ChartRequest(BaseModel):
    city: str
    chart_type: str


logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: open SQLite connection once
    base_dir = os.environ.get("OGD_BASE_DIR", ".")
    db_path = os.path.join(base_dir, db.DATABASE_FILENAME)
    logger.info("Connecting to sqlite DB at %s", db_path)

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


def period_default(period: str | None) -> str:
    if period:
        return period
    return charts.PERIOD_ALL


@app.exception_handler(StationNotFoundError)
async def station_not_found_handler(request, exc: StationNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


@app.exception_handler(NoDataError)
async def no_data_error_handler(request, exc: NoDataError):
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
    period: str | None = None,
    from_year: str | None = None,
    to_year: str | None = None,
    window: str | None = None,
):
    if chart_type not in charts.CHART_TYPE_COLUMNS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid chart type: {chart_type}",
        )
    period = period_default(period)

    station_abbr = station_abbr.upper()
    from_year_int = int(from_year) if from_year and from_year.isdigit() else None
    to_year_int = int(to_year) if to_year and to_year.isdigit() else None
    # Internal code treats window=None as "no window"
    window_int = int(window) if window and window.isdigit() and window != "1" else None

    df = db.read_daily_measurements(
        app.state.db,
        station_abbr,
        period=period,
        columns=charts.CHART_TYPE_COLUMNS[chart_type],
        from_year=from_year_int,
        to_year=to_year_int,
    )
    return {
        "vega_spec": charts.create_chart(
            chart_type, df, station_abbr, period=period, window=window_int
        ),
    }


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
    period: str | None = None,
    from_year: str | None = None,
    to_year: str | None = None,
    response: Response = None,
):
    period = period_default(period)
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

    # Stations don't change often, use 1 day TTL for caching.
    response.headers["Cache-Control"] = "public, max-age=86400"
    return {
        "summary": models.StationSummary(
            station=station,
            stats=stats,
        ),
    }


@app.get("/stations/{station_abbr}/info")
async def get_info(
    station_abbr: str,
    response: Response = None,
):
    station_abbr = station_abbr.upper()

    station = db.read_station(app.state.db, station_abbr)

    vars = db.read_station_var_summary_stats(
        app.state.db,
        agg_name=db.AGG_NAME_REF_1991_2020,
        station_abbr=station_abbr,
    )
    ref_period_stats = (
        charts.ref_period_stats(vars.loc[station_abbr]) if not vars.empty else None
    )

    # Stations don't change often, use 1 day TTL for caching.
    response.headers["Cache-Control"] = "public, max-age=86400"
    return {
        "info": models.StationInfo(
            station=station,
            ref_1991_2020_stats=ref_period_stats,
        ),
    }


@app.get("/stations")
async def list_stations(cantons: str | None = None, response: Response = None):
    cantons_list = None
    if cantons:
        cantons_list = cantons.split(",")

    stations = db.read_stations(app.state.db, cantons=cantons_list, exclude_empty=True)
    # Stations don't change often, use 1 day TTL for caching.
    response.headers["Cache-Control"] = "public, max-age=86400"
    return {
        "stations": stations,
    }
