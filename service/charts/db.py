import datetime
import logging
import os
from typing import Collection
import pandas as pd
from pydantic import BaseModel
import re
import sqlite3

from . import models
from .models import LocalizedString

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

TABLE_HOURLY_MEASUREMENTS = TableSpec(
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

TABLE_DAILY_MEASUREMENTS = TableSpec(
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
TABLE_NAME_META_STATIONS = "ogd_smn_meta_stations"
TABLE_NAME_META_PARAMETERS = "ogd_smn_meta_parameters"


# Derived tables / materialized views.
# To distinguish them from SoT data and mark them as derived,
# we prefix them all by "x_"
TABLE_NAME_X_STATION_DATA_SUMMARY = "ogd_smn_x_station_data_summary"
TABLE_NAME_X_STATION_VAR_SUMMARY_STATS = "ogd_smn_x_station_var_summary_stats"


# Aggregation names in STATION_VAR_SUMMARY_STATS_TABLE_NAME
AGG_NAME_REF_1991_2020 = "ref_1991_2020"


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
    base_dir: str,
    conn: sqlite3.Connection,
    table_spec: TableSpec,
    csv_filenames: list[str] | None = None,
) -> None:
    """Creates the table defined by table_spec if needed and inserts data from CSV files.

    Column names in the DB tables are identical to column names in the CSV (e.g.,
    station_abbr, reference_timestamp, tre200d0).

    Dates are stored as TEXT in ISO format "YYYY-MM-DD HH:MM:SSZ" and use UTC time.

    The time part is included only for sub-daily measurements, otherwise "YYYY-MM-DD" is used.
    """

    files = [os.path.join(base_dir, f) for f in csv_filenames]

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
        placeholders = ", ".join(["?"] * len(df.columns))
        insert_sql = f"INSERT OR IGNORE INTO {table_spec.name} VALUES ({placeholders})"

        c = conn.executemany(insert_sql, df.itertuples(index=False, name=None))
        logger.info(f"Inserted {c.rowcount} rows")
        conn.commit()


def recreate_station_var_summary_stats(conn: sqlite3.Connection) -> None:
    """(Re-)Creates a materialized view of summary data for the reference period 1991 to 2020."""
    # Drop existing data
    conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME_X_STATION_VAR_SUMMARY_STATS};")
    # Create table.
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME_X_STATION_VAR_SUMMARY_STATS} (
            agg_name TEXT,
            station_abbr TEXT,
            variable TEXT,
            source_granularity TEXT,
            min_value REAL,
            min_value_date TEXT,
            mean_value REAL,
            max_value REAL,
            max_value_date TEXT,
            source_count INTEGER,
            PRIMARY KEY (agg_name, station_abbr, variable)
        );
    """
    )
    # Insert data for aggregations (for now, just one).
    insert_ref_1991_2020_summary_stats(conn)


def insert_ref_1991_2020_summary_stats(conn: sqlite3.Connection) -> None:
    """Inserts summary stats for the 1991-2020 reference period into STATION_VAR_SUMMARY_STATS_TABLE_NAME."""
    variables = [
        TEMP_DAILY_MIN,
        TEMP_DAILY_MAX,
        TEMP_DAILY_MEAN,
        PRECIP_DAILY_MM,
        SUNSHINE_DAILY_MINUTES,
        WIND_SPEED_DAILY_MEAN,
        GUST_PEAK_DAILY_MAX,
        ATM_PRESSURE_DAILY_MEAN,
    ]
    sql = f"""
        SELECT
            station_abbr,
            reference_timestamp,
            {', '.join(variables)}
        FROM {TABLE_DAILY_MEASUREMENTS.name}
        WHERE reference_timestamp >= '1991-01-01' AND reference_timestamp < '2021-01-01'
    """
    df = pd.read_sql_query(
        sql,
        conn,
        dtype={
            "station_abbr": str,
            "reference_timestamp": str,
            **{v: float for v in variables},
        },
    )
    params = []
    for station_abbr, grp in df.groupby("station_abbr"):
        for var in variables:
            if grp[var].isna().all():
                continue  # No data for this variable
            a = grp[var].agg(["min", "max", "idxmin", "idxmax", "mean", "count"])
            params.append(
                (
                    station_abbr,
                    var,
                    "daily",
                    a["min"],
                    grp.loc[a["idxmin"], "reference_timestamp"],
                    a["mean"],
                    a["max"],
                    grp.loc[a["idxmax"], "reference_timestamp"],
                    a["count"],
                )
            )
    if params:
        insert_sql = f"""
            INSERT INTO {TABLE_NAME_X_STATION_VAR_SUMMARY_STATS} (
                agg_name,
                station_abbr,
                variable,
                source_granularity,
                min_value,
                min_value_date,
                mean_value,
                max_value,
                max_value_date,
                source_count
            )
            VALUES ('{AGG_NAME_REF_1991_2020}', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        conn.executemany(insert_sql, params)

    conn.commit()


def recreate_station_data_summary(conn: sqlite3.Connection) -> None:
    """Creates a materialized view of summary data per station_abbr.

    The summary stats in this table are calculated across the whole dataset.
    """
    # Drop old table
    conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME_X_STATION_DATA_SUMMARY};")

    # Create it with proper schema & PK
    conn.execute(
        f"""
        CREATE TABLE {TABLE_NAME_X_STATION_DATA_SUMMARY} (
            station_abbr TEXT PRIMARY KEY,
            station_name TEXT,
            station_canton TEXT,
            station_wigos_id TEXT,
            station_type_en TEXT,
            station_exposition_de TEXT,
            station_exposition_fr TEXT,
            station_exposition_it TEXT,
            station_exposition_en TEXT,
            station_url_de TEXT,
            station_url_fr TEXT,
            station_url_it TEXT,
            station_url_en TEXT,
            station_dataowner TEXT,
            station_data_since TEXT,
            station_height_masl REAL,
            station_coordinates_wgs84_lat REAL,
            station_coordinates_wgs84_lon REAL,

            tre200d0_count INTEGER NOT NULL,
            tre200d0_min_date TEXT,
            tre200d0_max_date TEXT,
            rre150d0_count INTEGER NOT NULL,
            rre150d0_min_date TEXT,
            rre150d0_max_date TEXT
        );
    """
    )

    conn.execute(
        f"""
        INSERT INTO {TABLE_NAME_X_STATION_DATA_SUMMARY}
        SELECT
            m.station_abbr,
            m.station_name,
            m.station_canton,
            m.station_wigos_id,
            m.station_type_en,
            m.station_exposition_de,
            m.station_exposition_fr,
            m.station_exposition_it,
            m.station_exposition_en,
            m.station_url_de,
            m.station_url_fr,
            m.station_url_it,
            m.station_url_en,
            m.station_dataowner,
            m.station_data_since,
            m.station_height_masl,
            m.station_coordinates_wgs84_lat,
            m.station_coordinates_wgs84_lon,

            COALESCE(h.tre200d0_count, 0),
            h.tre200d0_min_date,
            h.tre200d0_max_date,
            COALESCE(h.rre150d0_count, 0),
            h.rre150d0_min_date,
            h.rre150d0_max_date

        FROM {TABLE_NAME_META_STATIONS} AS m
        LEFT JOIN (
            SELECT
                station_abbr,
                SUM(CASE WHEN tre200d0 IS NOT NULL THEN 1 END) AS tre200d0_count,
                MIN(CASE WHEN tre200d0 IS NOT NULL THEN reference_timestamp END) AS tre200d0_min_date,
                MAX(CASE WHEN tre200d0 IS NOT NULL THEN reference_timestamp END) AS tre200d0_max_date,
                SUM(CASE WHEN rre150d0 IS NOT NULL THEN 1 END) AS rre150d0_count,
                MIN(CASE WHEN rre150d0 IS NOT NULL THEN reference_timestamp END) AS rre150d0_min_date,
                MAX(CASE WHEN rre150d0 IS NOT NULL THEN reference_timestamp END) AS rre150d0_max_date
            FROM {TABLE_DAILY_MEASUREMENTS.name}
            GROUP BY station_abbr
        ) AS h
        ON m.station_abbr = h.station_abbr;
    """
    )

    conn.commit()


def read_station(conn: sqlite3.Connection, station_abbr: str) -> models.Station:
    """Returns data for the given station."""

    sql = f"""
        SELECT
            station_abbr,
            station_name,
            station_canton,
            station_type_en,
            station_exposition_de,
            station_exposition_fr,
            station_exposition_it,
            station_exposition_en,
            station_url_de,
            station_url_fr,
            station_url_it,
            station_url_en,
            station_height_masl,
            station_coordinates_wgs84_lat,
            station_coordinates_wgs84_lon,
            tre200d0_min_date,
            tre200d0_max_date,
            rre150d0_min_date,
            rre150d0_max_date
        FROM {TABLE_NAME_X_STATION_DATA_SUMMARY}
        WHERE station_abbr = ?
    """
    cur = conn.execute(sql, [station_abbr])
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"No station found with abbr={station_abbr!r}")

    # Parse possible date strings into actual dates (None stays None)
    def d(v: str | None) -> datetime.date | None:
        return datetime.date.fromisoformat(v) if v else None

    return models.Station(
        abbr=row["station_abbr"],
        name=row["station_name"],
        canton=row["station_canton"],
        typ=row["station_type_en"],
        exposition=LocalizedString.from_nullable(
            de=row["station_exposition_de"],
            fr=row["station_exposition_fr"],
            it=row["station_exposition_it"],
            en=row["station_exposition_en"],
        ),
        url=LocalizedString.from_nullable(
            de=row["station_url_de"],
            fr=row["station_url_fr"],
            it=row["station_url_it"],
            en=row["station_url_en"],
        ),
        height_masl=row["station_height_masl"],
        coordinates_wgs84_lat=row["station_coordinates_wgs84_lat"],
        coordinates_wgs84_lon=row["station_coordinates_wgs84_lon"],
        temperature_min_date=d(row["tre200d0_min_date"]),
        temperature_max_date=d(row["tre200d0_max_date"]),
        precipitation_min_date=d(row["rre150d0_min_date"]),
        precipitation_max_date=d(row["rre150d0_max_date"]),
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
        SELECT 
            station_abbr,
            station_name,
            station_canton,
            station_type_en,
            station_exposition_de,
            station_exposition_fr,
            station_exposition_it,
            station_exposition_en,
            station_height_masl,
            station_coordinates_wgs84_lat,
            station_coordinates_wgs84_lon
        FROM {TABLE_NAME_X_STATION_DATA_SUMMARY}
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

    sql += " ORDER BY station_name"

    cur = conn.execute(sql, params)
    rows = cur.fetchall()

    return [
        models.Station(
            abbr=row["station_abbr"],
            name=row["station_name"],
            canton=row["station_canton"],
            typ=row["station_type_en"],
            exposition=LocalizedString.from_nullable(
                de=row["station_exposition_de"],
                fr=row["station_exposition_fr"],
                it=row["station_exposition_it"],
                en=row["station_exposition_en"],
            ),
            height_masl=row["station_height_masl"],
            coordinates_wgs84_lat=row["station_coordinates_wgs84_lat"],
            coordinates_wgs84_lon=row["station_coordinates_wgs84_lon"],
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
        FROM {TABLE_DAILY_MEASUREMENTS.name}
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
        # datetime.UTC is only available since Python 3.11
        d = d.astimezone(datetime.timezone.utc)
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
        FROM {TABLE_HOURLY_MEASUREMENTS.name}
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
        FROM {TABLE_NAME_META_PARAMETERS}
        ORDER BY parameter_shortname
    """
    df = pd.read_sql_query(sql, conn)
    df.columns = [c.removeprefix("parameter_") for c in df.columns]
    return df


def read_station_var_summary_stats(
    conn: sqlite3.Connection,
    agg_name: str,
    station_abbr: str | None = None,
    variables: Collection[str] | None = None,
) -> pd.DataFrame:
    """
    Reads summary stats for variables for the aggregation with the given `agg_name`.

    Usage examples:

        # Get summary values of a variable for a given station:
        df = read_station_var_summary_stats(conn, AGG_NAME_REF_1991_2020, "BER")
        p = df.loc["BER", "rre150dn"]
        print("Min precipitation level was", p["min_value"], "on", p["min_value_date"])

    Args:
        conn: An active SQLite database connection.
        agg_name: The name of the aggregataion to read data for.
        station_abbr: Optional abbreviation of the station to filter by.
        variables: Optional collection of variable names to filter by.

    Returns:
        A pandas DataFrame in wide format (rows indexed by 'station_abbr',
        columns as (variable, measurement_type)).
        Returns an empty DataFrame if no data matches the filters.
    """
    filters = ["agg_name = ?"]
    params = [agg_name]
    if station_abbr:
        filters.append("station_abbr = ?")
        params.append(station_abbr)
    if variables:
        placeholders = ", ".join(["?"] * len(variables))
        filters.append(f"variable IN ({placeholders})")
        params.extend(variables)
    measure_cols = [
        "source_granularity",
        "min_value",
        "min_value_date",
        "mean_value",
        "max_value",
        "max_value_date",
        "source_count",
    ]
    sql = f"""
        SELECT
            station_abbr,
            variable,
            {', '.join(measure_cols)}
        FROM {TABLE_NAME_X_STATION_VAR_SUMMARY_STATS}
    """
    if filters:
        sql += f"WHERE {' AND '.join(filters)}"

    df_long = pd.read_sql_query(sql, conn, params=params)

    if df_long.empty:
        return pd.DataFrame(index=pd.Index([], name="station_abbr"), dtype=str)

    return df_long.set_index(["station_abbr", "variable"]).unstack().swaplevel(axis=1)
