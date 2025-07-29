from contextlib import asynccontextmanager
import datetime
import logging
import os
import sqlalchemy as sa
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
    postgres_url = os.environ.get("OGD_POSTGRES_URL")

    if postgres_url:
        logger.info("Connecting to postgres DB at %s", postgres_url)
        engine = sa.create_engine(postgres_url, echo=False)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1"))
        logger.info("Successfully connected to postgres DB at %s", postgres_url)
    else:
        # Use SQLite database
        db_path = os.path.join(base_dir, db.DATABASE_FILENAME)
        logger.info("Connecting to sqlite DB at %s", db_path)
        engine = sa.create_engine(f"sqlite:///{db_path}", echo=True)

    app.state.engine = engine

    # If charts module needs to initialize something
    # charts.init_db(conn)

    yield

    logger.info("Shutting down")


# Always create the app, we're running this thing with uvicorn ONLY.
app = FastAPI(lifespan=lifespan)


def _period_default(period: str | None) -> str:
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
    period = _period_default(period)

    station_abbr = station_abbr.upper()
    from_year_int = int(from_year) if from_year and from_year.isdigit() else None
    to_year_int = int(to_year) if to_year and to_year.isdigit() else None
    # Internal code treats window=None as "no window"
    window_int = int(window) if window and window.isdigit() and window != "1" else None

    with app.state.engine.begin() as conn:
        df = db.read_daily_measurements(
            conn,
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


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=detail,
    )


def _daily_range(date: str) -> tuple[datetime.datetime, datetime.datetime]:
    try:
        d = datetime.date.fromisoformat(date)
    except ValueError:
        raise _bad_request("Invalid date: {date}")
    from_date = datetime.datetime(
        d.year, d.month, d.day, 1, 0, 0, 0, tzinfo=ZoneInfo("Europe/Zurich")
    )
    to_date = from_date + datetime.timedelta(days=1)
    return from_date, to_date


@app.get("/stations/{station_abbr}/charts/daily/{date}/{chart_type}")
async def get_daily_chart(
    station_abbr: str,
    date: str,
    chart_type: str,
):
    if chart_type not in ["overview"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Invalid chart type: {chart_type}",
        )

    from_date, to_date = _daily_range(date)
    station_abbr = station_abbr.upper()

    with app.state.engine.begin() as conn:
        df = db.read_hourly_measurements(
            conn,
            station_abbr,
            from_date=from_date,
            to_date=to_date,
            columns=[db.TEMP_HOURLY_MEAN, db.PRECIP_HOURLY_MM],
        )
    return {
        "vega_spec": charts.daily_temp_precip_chart(df, from_date, station_abbr),
    }


@app.get("/stations/{station_abbr}/daily")
async def get_daily_measurements(
    station_abbr: str,
    date: str | None = None,
):
    station_abbr = station_abbr.upper()
    # Parse date, if specified, else assume yesterday.
    from_date, to_date = _daily_range(date)

    with app.state.engine.begin() as conn:
        df = db.read_hourly_measurements(
            conn, station_abbr, from_date=from_date, to_date=to_date
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
    period = _period_default(period)
    station_abbr = station_abbr.upper()

    from_year_int = int(from_year) if from_year and from_year.isdigit() else None
    to_year_int = int(to_year) if to_year and to_year.isdigit() else None

    with app.state.engine.begin() as conn:
        df = db.read_daily_measurements(
            conn,
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
        station = db.read_station(conn, station_abbr)

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

    with app.state.engine.begin() as conn:
        station = db.read_station(conn, station_abbr)

        vars = db.read_station_var_summary_stats(
            conn,
            agg_name=db.AGG_NAME_REF_1991_2020,
            station_abbr=station_abbr,
        )
    ref_period_stats = (
        charts.station_period_stats(vars.loc[station_abbr]) if not vars.empty else None
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

    with app.state.engine.begin() as conn:
        stations = db.read_stations(conn, cantons=cantons_list, exclude_empty=True)
    # Stations don't change often, use 1 day TTL for caching.
    response.headers["Cache-Control"] = "public, max-age=86400"
    return {
        "stations": stations,
    }
