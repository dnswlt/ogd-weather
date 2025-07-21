import datetime
import re
import sqlite3
from typing import Callable, Iterable, TypeVar

import pandas as pd
from . import models

import sqlite3
from . import models

# Column names for temperature and precipitation measurements.
TEMP_HOURLY_MEAN = "tre200h0"
TEMP_HOURLY_MIN = "tre200hn"
TEMP_HOURLY_MAX = "tre200hx"
TEMP_DAILY_MEAN = "tre200d0"
TEMP_DAILY_MIN = "tre200dn"
TEMP_DAILY_MAX = "tre200dx"
PRECIP_HOURLY_MM = "rre150h0"
PRECIP_DAILY_MM = "rre150d0"


T = TypeVar("T")


def agg_none(func: Callable[[Iterable[T]], T], items: Iterable[T | None]) -> T | None:
    """Apply func (e.g. min or max) to items, ignoring None.
    Returns None if all values are None."""
    filtered = [x for x in items if x is not None]
    return func(filtered) if filtered else None


def read_station(conn: sqlite3.Connection, station_abbr: str) -> models.Station:
    """Returns data for the given station."""

    sql = """
        SELECT
            station_abbr,
            station_name,
            station_canton,
            tre200d0_min_date,
            tre200d0_max_date,
            rre150d0_min_date,
            rre150d0_max_date
        FROM ogd_smn_station_data_summary
        WHERE station_abbr = ?
    """
    cur = conn.execute(sql, [station_abbr])
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"No station found with abbr={station_abbr!r}")

    # Parse possible date strings into actual dates (None stays None)
    def d(key: str) -> datetime.date | None:
        v = row[key]
        return datetime.date.fromisoformat(v) if v else None

    # Pick overall min/max if any exist
    first_available_date = agg_none(
        min, [d("tre200d0_min_date"), d("rre150d0_min_date")]
    )
    last_available_date = agg_none(
        max, [d("tre200d0_max_date"), d("rre150d0_max_date")]
    )

    return models.Station(
        abbr=row["station_abbr"],
        name=row["station_name"],
        canton=row["station_canton"],
        first_available_date=first_available_date,
        last_available_date=last_available_date,
    )


def read_stations(
    conn: sqlite3.Connection,
    cantons: list[str] | None = None,
    exclude_empty: bool = True,
) -> list[models.Station]:
    """Returns all stations matching the given criteria.

    - cantons: optional list of canton codes to filter
    - exclude_empty: if True, skips stations with no temp/precip data
    """

    sql = """
        SELECT station_abbr, station_name, station_canton
        FROM ogd_smn_station_data_summary
    """
    filters = []
    params = []

    # Canton filter
    if cantons:
        placeholders = ",".join("?" for _ in cantons)
        filters.append(f"station_canton IN ({placeholders})")
        params.extend(cantons)

    # Exclude stations with no data
    if exclude_empty:
        filters.append("(tre200d0_count > 0 AND rre150d0_count > 0)")

    # Combine filters
    if filters:
        sql += " WHERE " + " AND ".join(filters)

    sql += " ORDER BY station_abbr"

    cur = conn.execute(sql, params)
    rows = cur.fetchall()

    return [
        models.Station(
            abbr=row["station_abbr"],
            name=row["station_name"],
            canton=row["station_canton"],
        )
        for row in rows
    ]


def read_daily_historical(
    conn: sqlite3.Connection,
    station_abbr: str,
    columns: list[str] | None = None,
    period: str | None = None,
    from_year: int | None = None,
    to_year: int | None = None,
) -> pd.DataFrame:
    if columns is None:
        columns = [TEMP_DAILY_MEAN, TEMP_DAILY_MIN, TEMP_DAILY_MAX, PRECIP_DAILY_MM]

    # Validate column names to prevent SQL injection
    if not all(re.search(r"^[a-z][a-zA-Z0-9_]*$", c) for c in columns):
        raise ValueError(f"Invalid columns: {','.join(columns)}")

    select_columns = ["station_abbr", "reference_timestamp"] + columns
    sql = f"""
        SELECT {', '.join(select_columns)}
        FROM ogd_smn_d_historical
    """
    # Filter by station.
    filters = ["station_abbr = ?"]
    params = [station_abbr]

    # Filter by period.
    if period is not None:
        if period.isdigit():
            filters.append(f"strftime('%m', reference_timestamp) = '{int(period):02d}'")
        elif period == "spring":
            filters.append("strftime('%m', reference_timestamp) IN ('03', '04', '05')")
        elif period == "summer":
            filters.append("strftime('%m', reference_timestamp) IN ('06', '07', '08')")
        elif period == "autumn":
            filters.append("strftime('%m', reference_timestamp) IN ('09', '10', '11')")
        elif period == "winter":
            filters.append("strftime('%m', reference_timestamp) IN ('12', '01', '02')")
        elif period == "all":
            pass  # No month filter

    if from_year is not None:
        filters.append(f"date(reference_timestamp) >= date('{from_year:04d}-01-01')")
    if to_year is not None:
        filters.append(f"date(reference_timestamp) < date('{to_year+1:04d}-01-01')")

    # Filter out any row that has only NULL measurements.
    if columns:
        non_null = " OR ".join(f"{c} IS NOT NULL" for c in columns)
        filters.append(f"({non_null})")

    sql += " WHERE " + " AND ".join(filters)
    sql += " ORDER BY reference_timestamp ASC"

    return pd.read_sql(
        sql,
        conn,
        params=params,
        parse_dates=["reference_timestamp"],
        index_col="reference_timestamp",
    )


def utc_timestr(d: datetime.datetime) -> str:
    """Returns the given datetime as a UTC time string in ISO format.

    Example: "2025-03-31 23:59:59Z"

    If d is a naive datetime (no tzinfo), it is assumed to be in UTC.
    """
    if d.tzinfo is not None:
        d = d.astimezone(datetime.UTC)
    return d.strftime("%Y-%m-%d %H:%M:%SZ")


def read_hourly_recent(
    conn: sqlite3.Connection,
    station_abbr: str,
    from_date: datetime.datetime,
    to_date: datetime.datetime,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if columns is None:
        columns = [TEMP_HOURLY_MEAN, TEMP_HOURLY_MIN, TEMP_HOURLY_MAX, PRECIP_HOURLY_MM]

    # Validate column names to prevent SQL injection
    if not all(re.search(r"^[a-z][a-zA-Z0-9_]*$", c) for c in columns):
        raise ValueError(f"Invalid columns: {','.join(columns)}")

    select_columns = ["station_abbr", "reference_timestamp"] + columns
    sql = f"""
        SELECT {', '.join(select_columns)}
        FROM ogd_smn_h_recent
    """
    # Filter by station.
    filters = ["station_abbr = ?"]
    params = [station_abbr]

    if from_date is not None:
        filters.append(f"reference_timestamp >= '{utc_timestr(from_date)}'")
    if to_date is not None:
        filters.append(f"reference_timestamp < '{utc_timestr(to_date)}'")

    # Filter out any row that has only NULL measurements.
    if columns:
        non_null = " OR ".join(f"{c} IS NOT NULL" for c in columns)
        filters.append(f"({non_null})")

    sql += " WHERE " + " AND ".join(filters)
    sql += " ORDER BY reference_timestamp ASC"

    return pd.read_sql(
        sql,
        conn,
        params=params,
        parse_dates=["reference_timestamp"],
        index_col="reference_timestamp",
    )
