import datetime
import glob
import logging
import os
import re
import sqlite3

import pandas as pd
from pydantic import BaseModel
from . import models

import sqlite3
from . import models

logger = logging.getLogger("db")


class TableSpec(BaseModel):
    """Specification for a measurement table."""

    name: str
    primary_key: list[str]
    measurements: list[str]
    date_format: dict[str, str] | None = None
    description: str = ""


# Filename for the sqlite3 database.
DATABASE_FILENAME = "swissmetnet.sqlite"

# Column names for weather measurements.

# Temperature
TEMP_HOURLY_MEAN = "tre200h0"
TEMP_HOURLY_MIN = "tre200hn"
TEMP_HOURLY_MAX = "tre200hx"
TEMP_DAILY_MEAN = "tre200d0"
TEMP_DAILY_MIN = "tre200dn"
TEMP_DAILY_MAX = "tre200dx"

# Precipitation
PRECIP_HOURLY_MM = "rre150h0"
PRECIP_DAILY_MM = "rre150d0"

# Wind
WIND_SPEED_DAILY_MEAN = "fkl010d0"
WIND_SPEED_HOURLY_MEAN = "fkl010h0"
WIND_DIRECTION_DAILY_MEAN = "dkl010d0"
WIND_DIRECTION_HOURLY_MEAN = "dkl010h0"
GUST_PEAK_DAILY_MAX = "fkl010d1"
GUST_PEAK_HOURLY_MAX = "fkl010h1"

# Atmospheric pressure
ATM_PRESSURE_DAILY_MEAN = "prestad0"
ATM_PRESSURE_HOURLY_MEAN = "prestah0"

# Humidity
REL_HUMITIDY_DAILY_MEAN = "ure200d0"
REL_HUMITIDY_HOURLY_MEAN = "ure200h0"

# Sunshine
SUNSHINE_DAILY_MINUTES = "sre000d0"
SUNSHINE_HOURLY_MINUTES = "sre000h0"


# Table definitions

HOURLY_MEASUREMENTS_TABLE = TableSpec(
    name="ogd_smn_hourly",
    primary_key=[
        "station_abbr",
        "reference_timestamp",
    ],
    date_format={
        "reference_timestamp": "%d.%m.%Y %H:%M",
    },
    measurements=[
        TEMP_HOURLY_MEAN,
        TEMP_HOURLY_MIN,
        TEMP_HOURLY_MAX,
        PRECIP_HOURLY_MM,
        GUST_PEAK_HOURLY_MAX,
        ATM_PRESSURE_HOURLY_MEAN,
        REL_HUMITIDY_HOURLY_MEAN,
        SUNSHINE_HOURLY_MINUTES,
        WIND_SPEED_HOURLY_MEAN,
        WIND_DIRECTION_HOURLY_MEAN,
    ],
)

DAILY_MEASUREMENTS_TABLE = TableSpec(
    name="ogd_smn_daily",
    primary_key=[
        "station_abbr",
        "reference_timestamp",
    ],
    date_format={
        "reference_timestamp": "%d.%m.%Y %H:%M",
    },
    measurements=[
        TEMP_DAILY_MEAN,
        TEMP_DAILY_MIN,
        TEMP_DAILY_MAX,
        PRECIP_DAILY_MM,
        GUST_PEAK_DAILY_MAX,
        ATM_PRESSURE_DAILY_MEAN,
        REL_HUMITIDY_DAILY_MEAN,
        SUNSHINE_DAILY_MINUTES,
        WIND_SPEED_DAILY_MEAN,
        WIND_DIRECTION_DAILY_MEAN,
    ],
)

# Metadata tables
META_STATIONS_TABLE_NAME = "ogd_smn_meta_stations"
META_PARAMETERS_TABLE_NAME = "ogd_smn_meta_parameters"


# Derived tables / materialized views
STATION_DATA_SUMMARY_TABLE_NAME = "ogd_smn_station_data_summary"


def agg_none(func, items):
    """Apply func (e.g. min or max) to items, ignoring None.
    Returns None if all values are None."""
    filtered = [x for x in items if x is not None]
    return func(filtered) if filtered else None


def normalize_timestamp(series: pd.Series) -> pd.Series:
    """Normalizes a datetime Series for DB storage.

    - If all times are midnight, returns only "YYYY-MM-DD".
    - Otherwise assumes UTC, returns ISO with second granularity.
    """
    if series.empty:
        return series

    time_of_day = series.dt.time

    if time_of_day.nunique() == 1 and time_of_day.iloc[0] == datetime.time(0, 0):
        return series.dt.strftime("%Y-%m-%d")
    else:
        # Sub-daily measurements: use UTC
        return series.dt.tz_localize("UTC").dt.strftime("%Y-%m-%d %H:%M:%SZ")


def prepare_sql_table_from_spec(
    dir: str,
    conn: sqlite3.Connection,
    table_spec: TableSpec,
    csv_pattern: str,
) -> None:
    """Creates the table defined by table_spec if needed and inserts data from CSV files."""

    files = glob.glob(os.path.join(dir, csv_pattern))
    if not files:
        logger.info(f"No files match pattern {csv_pattern}")
        return

    # Create table if needed.
    pk_columns = ", ".join(table_spec.primary_key)
    columns = ",\n".join(
        [f"{k} TEXT" for k in table_spec.primary_key]
        + [f"{k} REAL" for k in table_spec.measurements]
    )
    sql = f"""
        CREATE TABLE IF NOT EXISTS {table_spec.name} (
            {columns},
            PRIMARY KEY ({pk_columns})
        )
        """
    conn.execute(sql)
    conn.commit()

    csv_dtype = {}
    for c in table_spec.measurements:
        csv_dtype[c] = float

    columns = table_spec.primary_key + table_spec.measurements
    for fname in files:
        logger.info(f"Importing {fname} into sqlite3 database...")
        df = pd.read_csv(
            fname,
            sep=";",
            encoding="cp1252",
            date_format=table_spec.date_format,
            parse_dates=list(table_spec.date_format.keys()),
            dtype=csv_dtype,
            usecols=columns,
        )
        df = df[columns]  # Reorder columns to match the order in table_spec.
        # Normalize timestamps: use YYYY-MM-DD if time is always 00:00
        for col in table_spec.date_format.keys():
            df[col] = normalize_timestamp(df[col])

        # Insert row by row, ignoring duplicates
        placeholders = ",".join(["?"] * len(df.columns))
        insert_sql = f"INSERT OR IGNORE INTO {table_spec.name} VALUES ({placeholders})"

        c = conn.executemany(insert_sql, df.itertuples(index=False, name=None))
        logger.info(f"Inserted {c.rowcount} rows")
        conn.commit()


def read_station(conn: sqlite3.Connection, station_abbr: str) -> models.Station:
    """Returns data for the given station."""

    sql = f"""
        SELECT
            station_abbr,
            station_name,
            station_canton,
            tre200d0_min_date,
            tre200d0_max_date,
            rre150d0_min_date,
            rre150d0_max_date
        FROM {STATION_DATA_SUMMARY_TABLE_NAME}
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

    sql = f"""
        SELECT station_abbr, station_name, station_canton
        FROM {STATION_DATA_SUMMARY_TABLE_NAME}
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


def read_daily_measurements(
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
        FROM {DAILY_MEASUREMENTS_TABLE.name}
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

    return pd.read_sql_query(
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


def read_hourly_measurements(
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
        FROM {HOURLY_MEASUREMENTS_TABLE.name}
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

    return pd.read_sql_query(
        sql,
        conn,
        params=params,
        parse_dates=["reference_timestamp"],
        index_col="reference_timestamp",
    )


def read_parameters(conn: sqlite3.Connection) -> pd.DataFrame:
    sql = f"""
        SELECT
            parameter_shortname,
            parameter_description_de,
            parameter_description_fr,
            parameter_description_it,
            parameter_description_en,
            parameter_group_de,
            parameter_group_fr,
            parameter_group_it,
            parameter_group_en,
            parameter_datatype,
            parameter_unit
        FROM {META_PARAMETERS_TABLE_NAME}
        ORDER BY parameter_shortname
    """
    df = pd.read_sql_query(sql, conn)
    df.columns = [c.removeprefix("parameter_") for c in df.columns]
    return df
