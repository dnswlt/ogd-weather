from contextlib import asynccontextmanager
import datetime
from io import StringIO
import logging
import os
from urllib.parse import urlparse, urlunparse
import sqlalchemy as sa
from fastapi import FastAPI, HTTPException, Request, status, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from zoneinfo import ZoneInfo
from . import charts
from . import db
from . import models
from . import logging_config as _  # configure logging
from .errors import NoDataError, StationNotFoundError


logger = logging.getLogger("app")


def _pg_user(conn_str: str) -> str:
    """Returns the user of the given postgres connection URL."""
    return urlparse(conn_str).username


def _sanitize_pg_url(conn_str: str) -> str:
    """Returns a connection URL for postgres with the password removed (if present)."""
    if not conn_str:
        return conn_str

    parsed = urlparse(conn_str)
    if parsed.username is None:
        return conn_str  # no user info, nothing to redact

    # Rebuild netloc without password
    netloc = parsed.hostname or ""
    if parsed.username:
        netloc = parsed.username + "@" + netloc
    if parsed.port:
        netloc += f":{parsed.port}"

    sanitized = parsed._replace(netloc=netloc)
    return urlunparse(sanitized)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: open SQLite connection once
    base_dir = os.environ.get("OGD_BASE_DIR", ".")
    postgres_url = os.environ.get("OGD_POSTGRES_URL", "")

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

    app.state.server_options = models.ServerOptions(
        base_dir=base_dir,
        sanitized_postgres_url=_sanitize_pg_url(postgres_url),
        start_time=datetime.datetime.now(tz=datetime.timezone.utc),
    )
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


def _vega_chart(request: Request, chart: charts.AltairChart) -> Response:
    if "text/html" in request.headers.get("accept", ""):
        sb = StringIO()
        chart.save(sb, format="html")
        return HTMLResponse(content=sb.getvalue())

    return JSONResponse(
        content={
            "vega_spec": chart.to_dict(),
        }
    )


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


@app.get("/status")
def server_status():
    """Returns status informatin for the running server."""
    status = models.ServerStatus(
        current_time_utc=datetime.datetime.now(tz=datetime.timezone.utc),
        db_engine=app.state.engine.name,
        options=app.state.server_options,
    )

    if app.state.engine.name == "postgresql":
        user = _pg_user(app.state.server_options.sanitized_postgres_url)
        status.db_table_stats = db.table_stats(app.state.engine, user=user)

    return status


@app.get("/stations/{station_abbr}/charts/annual/{chart_type}")
async def get_annual_chart(
    request: Request,
    station_abbr: str,
    chart_type: str,
    period: str | None = None,
    from_year: str | None = None,
    to_year: str | None = None,
    window: str | None = None,
):
    if chart_type not in charts.CHART_TYPE_COLUMNS:
        raise _bad_request(f"Invalid chart type: {chart_type}")

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

    chart = charts.create_annual_chart(
        chart_type, df, station_abbr, period=period, window=window_int
    )
    return _vega_chart(request, chart)


@app.get("/stations/{station_abbr}/charts/year/{year}/{chart_type}")
async def get_year_chart(
    request: Request,
    station_abbr: str,
    year: int,
    chart_type: str,
):

    station_abbr = station_abbr.upper()

    def _read_data(columns):
        with app.state.engine.begin() as conn:
            return db.read_daily_measurements(
                conn,
                station_abbr,
                period="all",
                columns=columns,
                from_year=year,
                to_year=year,
            )

    if chart_type == "drywet":
        df = _read_data([db.PRECIP_DAILY_MM, db.SUNSHINE_DAILY_PCT_OF_MAX])
        chart = charts.drywet_grid_chart(df, station_abbr, year)
    elif chart_type == "temperature:month":
        df = _read_data([db.TEMP_DAILY_MAX])
        chart = charts.monthly_temp_boxplot_chart(df, station_abbr, year)
    elif chart_type == "sunshine:month":
        df = _read_data([db.SUNSHINE_DAILY_MINUTES])
        chart = charts.monthly_sunshine_boxplot_chart(df, station_abbr, year)
    elif chart_type == "humidity:month":
        df = _read_data([db.REL_HUMITIDY_DAILY_MEAN])
        chart = charts.monthly_humidity_boxplot_chart(df, station_abbr, year)
    else:
        raise _bad_request(f"Invalid chart type: {chart_type}")

    return _vega_chart(request, chart)


@app.get("/stations/{station_abbr}/charts/day/{date}/{chart_type}")
async def get_daily_chart(
    request: Request,
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
    chart = charts.daily_temp_precip_chart(df, from_date, station_abbr)

    return _vega_chart(request, chart)


@app.get("/stations/{station_abbr}/stats/day/{date}/measurements")
async def get_daily_measurements(
    station_abbr: str,
    date: str,
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


def _dt_range(
    from_date: str, to_date: str, date: str
) -> tuple[datetime.datetime, datetime.datetime]:
    if date:
        return _daily_range(date)
    if from_date == "" and to_date == "":
        # No parameter set: assume 2daysago
        date = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime(
            "%Y-%m-%d"
        )
        return _daily_range(date)
    try:
        d1 = datetime.date.fromisoformat(from_date)
        from_dt = datetime.datetime(
            d1.year, d1.month, d1.day, 0, 0, 0, 0, tzinfo=ZoneInfo("Europe/Zurich")
        )
        d2 = datetime.date.fromisoformat(to_date)
        to_dt = datetime.datetime(
            d2.year, d2.month, d2.day, 0, 0, 0, 0, tzinfo=ZoneInfo("Europe/Zurich")
        ) + datetime.timedelta(days=1)
        return (from_dt, to_dt)
    except Exception:
        raise _bad_request(f"Invalid from_date or to_date: '{from_date}', '{to_date}'")


@app.get("/stations/{station_abbr}/data/{granularity}")
async def get_data(
    station_abbr: str,
    granularity: str,
    from_date: str = "",
    to_date: str = "",
    date: str = "",
    limit: str = "",
    csv: bool = False,
):
    station_abbr = station_abbr.upper()

    if granularity not in ("hourly"):
        raise _bad_request(f"Invalid granularity: {granularity}")
    # if format not in ("text/csv", "csv", "application/json", "json"):
    #     raise _bad_request(f"Invalid format: {format}")

    from_dt, to_dt = _dt_range(from_date, to_date, date)

    limit_int = int(limit) if limit.isdigit() else 1000
    if limit_int <= 0 or limit_int > 1000:
        raise _bad_request(
            f"Invalid limit (limit={limit}, min=1, max=1000). This endpoint is not meant as a data dump."
        )
    with app.state.engine.begin() as conn:
        df = db.read_hourly_measurements(
            conn,
            station_abbr,
            from_date=from_dt,
            to_date=to_dt,
            limit=limit_int,
        )

    # Use Europe/Zurich local time in export. To avoid pandas time formatting
    # getting in the way, format as string up front explicitly.
    df.index = df.index.tz_convert("Europe/Zurich").strftime("%Y-%m-%d %H:%M:%S")
    df = df.reset_index()
    if csv:
        buf = StringIO()
        df.to_csv(buf, sep=",", header=True, index=False, encoding="utf-8")
        return Response(
            content=buf.getvalue(),
            media_type="text/csv; charset=utf-8",
        )

    return Response(
        content=df.to_json(orient="records", index=False),
        media_type="application/json",
    )


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
