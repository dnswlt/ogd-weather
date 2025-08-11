import datetime
import io
import itertools
import logging
import os
import time
from typing import Any, Callable, Collection
from urllib.parse import urlparse
import uuid
import numpy as np
import pandas as pd
from pydantic import BaseModel
import re
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from .pandas_funcs import pctl
from . import geo
from . import models
from . import sql_queries

from .models import LocalizedString
from .errors import StationNotFoundError

logger = logging.getLogger("db")


# SQLAlchemy pattern: have a global 'metadata' variable that holds all table defs.
metadata = sa.MetaData()


class UpdateStatus(BaseModel):
    """Represents a row in the update_status tracking table."""

    id: str | None  # UUID-4, empty if this is a new record
    href: str
    table_updated_time: datetime.datetime
    resource_updated_time: datetime.datetime | None = None
    etag: str | None = None

    def filename(self):
        """Returns the filename part of href."""
        return os.path.basename(urlparse(self.href).path)


class DataTableSpec:
    """Schema specification for a measurement table + SQLAlchmey Table binding."""

    def __init__(
        self,
        name: str,
        primary_key: list[str],
        measurements: list[str],
        date_format: dict[str, str] | None = None,
        description: str = "",
    ) -> None:
        self.name: str = name
        self.primary_key: list[str] = primary_key
        self.measurements: list[str] = measurements
        self.date_format: dict[str, str] | None = date_format
        self.description: str = description
        # Reference to the sa.Table instance:
        # Ensure this table gets registered in metadata immediately.
        self.sa_table: sa.Table = self._define_sa_table()

    def _define_sa_table(self) -> sa.Table:
        """Converts a DataTableSpec into a SQLAlchemy Core Table definition.

        Registers this table in the global `metadata`.
        """

        # Primary key columns are always TEXT
        pk_columns = [
            sa.Column(col_name, sa.Text, primary_key=True)
            for col_name in self.primary_key
        ]

        # Measurement columns are always REAL: Use 32-bit accuracy is more than enough.
        measurement_columns = [
            sa.Column(col_name, sa.REAL) for col_name in self.measurements
        ]

        return sa.Table(self.name, metadata, *(pk_columns + measurement_columns))

    def columns(self):
        return self.primary_key + self.measurements


# Filename for the sqlite3 database.
DATABASE_FILENAME = "swissmetnet.sqlite"

# Column names for weather measurements.

# Temperature
TEMP_HOURLY_MIN = "tre200hn"
TEMP_HOURLY_MEAN = "tre200h0"
TEMP_HOURLY_MAX = "tre200hx"
TEMP_DAILY_MIN = "tre200dn"
TEMP_DAILY_MEAN = "tre200d0"
TEMP_DAILY_MAX = "tre200dx"
TEMP_MONTHLY_MIN = "tre200mn"
TEMP_MONTHLY_MEAN = "tre200m0"
TEMP_MONTHLY_MAX = "tre200mx"

# Precipitation
PRECIP_HOURLY_MM = "rre150h0"
PRECIP_DAILY_MM = "rre150d0"
PRECIP_MONTHLY_MM = "rre150m0"

# Wind
WIND_SPEED_HOURLY_MEAN = "fkl010h0"
WIND_SPEED_DAILY_MEAN = "fkl010d0"
WIND_SPEED_MONTHLY_MEAN = "fkl010m0"

WIND_DIRECTION_HOURLY_MEAN = "dkl010h0"
WIND_DIRECTION_DAILY_MEAN = "dkl010d0"

GUST_PEAK_HOURLY_MAX = "fkl010h1"
GUST_PEAK_DAILY_MAX = "fkl010d1"
GUST_PEAK_MONTHLY_MAX = "fkl010m1"

# Atmospheric pressure
ATM_PRESSURE_DAILY_MEAN = "prestad0"
ATM_PRESSURE_HOURLY_MEAN = "prestah0"
ATM_PRESSURE_MONTHLY_MEAN = "prestam0"

# Humidity
REL_HUMITIDY_HOURLY_MEAN = "ure200h0"
REL_HUMITIDY_DAILY_MEAN = "ure200d0"
REL_HUMITIDY_MONTHLY_MEAN = "ure200m0"

# Sunshine
SUNSHINE_HOURLY_MINUTES = "sre000h0"
SUNSHINE_DAILY_MINUTES = "sre000d0"
SUNSHINE_MONTHLY_MINUTES = "sre000m0"

SUNSHINE_DAILY_PCT_OF_MAX = "sremaxdv"

# Snow
SNOW_DEPTH_DAILY_CM = "htoautd0"

# Map SwissMetNet parameter names to readable names to use at the API level
# (e.g. when returning summary stats for variables).
VARIABLE_API_NAMES = {
    TEMP_DAILY_MIN: "temperature_daily_min",
    TEMP_DAILY_MAX: "temperature_daily_max",
    TEMP_DAILY_MEAN: "temperature_daily_mean",
    PRECIP_DAILY_MM: "precipitation_daily_millimeters",
    WIND_SPEED_DAILY_MEAN: "wind_speed_daily_mean",
    WIND_DIRECTION_DAILY_MEAN: "wind_direction_daily_mean",
    SUNSHINE_DAILY_MINUTES: "sunshine_daily_minutes",
    GUST_PEAK_DAILY_MAX: "gust_peak_daily_max",
    ATM_PRESSURE_DAILY_MEAN: "atm_pressure_daily_mean",
    REL_HUMITIDY_DAILY_MEAN: "rel_humidity_daily_mean",
    SUNSHINE_DAILY_MINUTES: "sunshine_daily_minutes",
    SUNSHINE_DAILY_PCT_OF_MAX: "sunshine_daily_pct_of_max",
    SNOW_DEPTH_DAILY_CM: "snow_depth_daily_cm",
}

# Derived metric names (prefix DX_):
DX_SUNNY_DAYS_ANNUAL_COUNT = "sunny_days_annual_count"
DX_SUMMER_DAYS_ANNUAL_COUNT = "summer_days_annual_count"
DX_FROST_DAYS_ANNUAL_COUNT = "frost_days_annual_count"
DX_TROPICAL_NIGHTS_ANNUAL_COUNT = "tropical_nights_annual_count"
DX_GROWING_DEGREE_DAYS_ANNUAL_SUM = "growing_degree_days_annual_sum"

DX_SUNNY_DAYS_MONTHLY_COUNT = "sunny_days_monthly_count"
DX_SUMMER_DAYS_MONTHLY_COUNT = "summer_days_monthly_count"
DX_FROST_DAYS_MONTHLY_COUNT = "frost_days_monthly_count"
DX_TROPICAL_NIGHTS_MONTHLY_COUNT = "tropical_nights_monthly_count"
DX_GROWING_DEGREE_DAYS_MONTHLY_SUM = "growing_degree_days_monthly_sum"

DX_SOURCE_DATE_RANGE = "source_date_range"

# Table definitions

TABLE_HOURLY_MEASUREMENTS = DataTableSpec(
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

TABLE_DAILY_MEASUREMENTS = DataTableSpec(
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
        SUNSHINE_DAILY_PCT_OF_MAX,
        WIND_SPEED_DAILY_MEAN,
        WIND_DIRECTION_DAILY_MEAN,
        SNOW_DEPTH_DAILY_CM,
    ],
)

TABLE_MONTHLY_MEASUREMENTS = DataTableSpec(
    name="ogd_smn_monthly",
    primary_key=[
        "station_abbr",
        "reference_timestamp",
    ],
    date_format={
        "reference_timestamp": "%d.%m.%Y %H:%M",
    },
    measurements=[
        TEMP_MONTHLY_MIN,
        TEMP_MONTHLY_MEAN,
        TEMP_MONTHLY_MAX,
        PRECIP_MONTHLY_MM,
        WIND_SPEED_MONTHLY_MEAN,
        GUST_PEAK_MONTHLY_MAX,
        ATM_PRESSURE_MONTHLY_MEAN,
        REL_HUMITIDY_MONTHLY_MEAN,
        SUNSHINE_MONTHLY_MINUTES,
    ],
)


sa_table_meta_stations = sa.Table(
    "ogd_smn_meta_stations",
    metadata,
    sa.Column("station_abbr", sa.Text, primary_key=True),
    sa.Column("station_name", sa.Text),
    sa.Column("station_canton", sa.Text),
    sa.Column("station_wigos_id", sa.Text),
    sa.Column("station_type_de", sa.Text),
    sa.Column("station_type_fr", sa.Text),
    sa.Column("station_type_it", sa.Text),
    sa.Column("station_type_en", sa.Text),
    sa.Column("station_dataowner", sa.Text),
    sa.Column("station_data_since", sa.Text),
    sa.Column("station_height_masl", sa.REAL),
    sa.Column("station_height_barometer_masl", sa.REAL),
    sa.Column("station_coordinates_lv95_east", sa.REAL),
    sa.Column("station_coordinates_lv95_north", sa.REAL),
    sa.Column("station_coordinates_wgs84_lat", sa.REAL),
    sa.Column("station_coordinates_wgs84_lon", sa.REAL),
    sa.Column("station_exposition_de", sa.Text),
    sa.Column("station_exposition_fr", sa.Text),
    sa.Column("station_exposition_it", sa.Text),
    sa.Column("station_exposition_en", sa.Text),
    sa.Column("station_url_de", sa.Text),
    sa.Column("station_url_fr", sa.Text),
    sa.Column("station_url_it", sa.Text),
    sa.Column("station_url_en", sa.Text),
)

sa_table_meta_parameters = sa.Table(
    "ogd_smn_meta_parameters",
    metadata,
    sa.Column("parameter_shortname", sa.Text, primary_key=True),
    sa.Column("parameter_description_de", sa.Text),
    sa.Column("parameter_description_fr", sa.Text),
    sa.Column("parameter_description_it", sa.Text),
    sa.Column("parameter_description_en", sa.Text),
    sa.Column("parameter_group_de", sa.Text),
    sa.Column("parameter_group_fr", sa.Text),
    sa.Column("parameter_group_it", sa.Text),
    sa.Column("parameter_group_en", sa.Text),
    sa.Column("parameter_granularity", sa.Text),
    sa.Column("parameter_decimals", sa.Integer),
    sa.Column("parameter_datatype", sa.Text),
    sa.Column("parameter_unit", sa.Text),
)


sa_table_update_status = sa.Table(
    "update_status",
    metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("href", sa.Text, unique=True, nullable=False),
    sa.Column("table_updated_time", sa.Text, nullable=False),
    sa.Column("resource_updated_time", sa.Text),
    sa.Column("etag", sa.Text),
)


# Derived tables / materialized views.
# To distinguish them from SoT data and mark them as derived,
# we prefix them all by "x_"

sa_table_x_station_data_summary = sa.Table(
    "x_station_data_summary",
    metadata,
    sa.Column("station_abbr", sa.Text, primary_key=True),
    sa.Column("station_name", sa.Text),
    sa.Column("station_canton", sa.Text),
    sa.Column("station_wigos_id", sa.Text),
    sa.Column("station_type_en", sa.Text),
    sa.Column("station_exposition_de", sa.Text),
    sa.Column("station_exposition_fr", sa.Text),
    sa.Column("station_exposition_it", sa.Text),
    sa.Column("station_exposition_en", sa.Text),
    sa.Column("station_url_de", sa.Text),
    sa.Column("station_url_fr", sa.Text),
    sa.Column("station_url_it", sa.Text),
    sa.Column("station_url_en", sa.Text),
    sa.Column("station_dataowner", sa.Text),
    sa.Column("station_data_since", sa.Text),
    sa.Column("station_height_masl", sa.REAL),
    sa.Column("station_coordinates_wgs84_lat", sa.REAL),
    sa.Column("station_coordinates_wgs84_lon", sa.REAL),
    sa.Column("tre200d0_count", sa.Integer, nullable=False),
    sa.Column("tre200d0_min_date", sa.Text),
    sa.Column("tre200d0_max_date", sa.Text),
    sa.Column("rre150d0_count", sa.Integer, nullable=False),
    sa.Column("rre150d0_min_date", sa.Text),
    sa.Column("rre150d0_max_date", sa.Text),
)

sa_table_x_nearby_stations = sa.Table(
    "x_nearby_stations",
    metadata,
    sa.Column("from_station_abbr", sa.Text, primary_key=True),
    sa.Column("from_station_name", sa.Text),
    sa.Column("from_station_canton", sa.Text),
    sa.Column("to_station_abbr", sa.Text, primary_key=True),
    sa.Column("to_station_name", sa.Text),
    sa.Column("to_station_canton", sa.Text),
    sa.Column("distance_km", sa.REAL),
    sa.Column("height_diff", sa.REAL),
)


# Aggregation names in STATION_VAR_SUMMARY_STATS_TABLE_NAME
AGG_NAME_REF_1991_2020 = "ref_1991_2020"

# Time slice value for aggregations that have no actual time slice.
TS_ALL = "*"


class VarSummaryStatsTable:

    def __init__(self, name: str, time_slice: str) -> None:
        self.name = name
        if time_slice == "all":
            self.time_slicer = VarSummaryStatsTable._ts_all
        elif time_slice == "month":
            self.time_slicer = VarSummaryStatsTable._ts_month
        else:
            raise ValueError(f"invalid time slice: {time_slice}")
        self.sa_table: sa.Table = self._define_station_var_summary_stats_table(name)

    def _define_station_var_summary_stats_table(self, table_name: str) -> sa.Table:
        return sa.Table(
            table_name,
            metadata,
            sa.Column("agg_name", sa.Text, primary_key=True),
            sa.Column("station_abbr", sa.Text, primary_key=True),
            sa.Column("variable", sa.Text, primary_key=True),
            sa.Column("time_slice", sa.Text, primary_key=True),
            sa.Column("source_granularity", sa.Text),
            # Min/max including date of occurrence
            sa.Column("min_value", sa.REAL),
            sa.Column("min_value_date", sa.Text),
            sa.Column("mean_value", sa.REAL),
            sa.Column("max_value", sa.REAL),
            sa.Column("max_value_date", sa.Text),
            # Percentiles (10, 25, 50, 75, 90)
            sa.Column("p10_value", sa.REAL),
            sa.Column("p25_value", sa.REAL),
            sa.Column("median_value", sa.REAL),
            sa.Column("p75_value", sa.REAL),
            sa.Column("p90_value", sa.REAL),
            # Sum
            sa.Column("value_sum", sa.REAL),
            # Count
            sa.Column("value_count", sa.Integer),
        )

    @classmethod
    def _ts_month(cls, d: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(d["reference_timestamp"]).dt.month.map(ts_month)

    @classmethod
    def _ts_all(cls, d: pd.DataFrame) -> pd.Series:
        return pd.Series(["*"] * len(d.index), index=d.index)


var_summary_stats_all = VarSummaryStatsTable(
    name="x_station_var_summary_stats",
    time_slice="all",
)


var_summary_stats_month = VarSummaryStatsTable(
    name="x_station_var_summary_stats_month",
    time_slice="month",
)


def ts_month(month: int) -> str:
    """Returns the time_slice string for the given month."""
    return f"{month:02d}"


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


def insert_csv_metadata(
    base_dir: str,
    engine: sa.Engine,
    table: sa.Table,
    update: UpdateStatus,
    drop_existing: bool = True,
) -> None:
    """Loads CSV and insert rows into an existing metadata table.

    Existing rows in the metadata table are dropped if `drop_existing` is True.

    - Requires all table columns to be present in the CSV.
    """

    # Columns defined in the SQLAlchemy table (in order)
    table_columns = [col.name for col in table.columns]

    # Load CSV
    csv_file = update.filename()
    df = pd.read_csv(os.path.join(base_dir, csv_file), sep=";", encoding="cp1252")

    # Validate all columns present
    missing = [c for c in table_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_file} is missing required columns: {missing}")

    # Select only required columns, in correct order
    df = df[table_columns]

    with engine.begin() as conn:
        if drop_existing:
            conn.execute(sa.delete(table))
        records = df.replace({np.nan: None}).to_dict(orient="records")
        conn.execute(sa.insert(table), records)
        # Update status
        save_update_status(conn, update)


def insert_csv_data(
    base_dir: str,
    engine: sa.Engine,
    table_spec: DataTableSpec,
    update: UpdateStatus,
    insert_mode: str = "insert_missing",
) -> None:
    """Inserts data from a CSV file specified in `update`.

    *  Column names in the DB tables are identical to column names in the CSV (e.g.,
    station_abbr, reference_timestamp, tre200d0).

    *  Dates are stored as TEXT in ISO format "YYYY-MM-DD HH:MM:SSZ" and use UTC time.

    *  Measurements are stored as REAL (32 bit).

    *  The time part is included only for sub-daily measurements, otherwise "YYYY-MM-DD" is used.

    *  The UpdateStatus DB table is updated in the same transaction.

    :param base_dir:
    The directory in which CSV files are expected.

    :param engine:
    The SQLAlchemy engine.

    :param table_spec:
    The table to insert/upsert data into.

    :param update:
    Information about the CSV file to insert. Also used to update the status_update table.

    :param mode:
    The insert mode; must be one of

        * "append" - Appends all CSV rows directly to the destination table.
            This ignores existing rows and will fail if duplicate primary keys
            are found.
        * "merge" - Inserts or updates all CSV rows. Existing rows with the
            same primary key will be updated.
        * "insert_missing" - Only inserts CSV rows whose primary keys do not exist
            in the table yet.
    """

    if insert_mode not in ("append", "merge", "insert_missing"):
        raise ValueError(f"Invalid mode: {insert_mode}")

    filename = os.path.join(base_dir, update.filename())

    csv_dtype = {}
    for c in table_spec.measurements:
        csv_dtype[c] = float

    columns = table_spec.columns()
    logger.info(f"Importing {filename} into database...")
    df = pd.read_csv(
        filename,
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

    start_time = time.time()
    # Insert rows, ignoring duplicates
    with engine.begin() as conn:
        if insert_mode == "merge":
            if engine.name != "postgresql":
                raise ValueError("Merge is only supported for postgresql")
            # Use ON CONFLICT DO UPDATE for PostgreSQL:
            insert_stmt = postgresql.insert(table_spec.sa_table)
            insert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=table_spec.sa_table.primary_key,
                set_={
                    c: insert_stmt.excluded[c]
                    for c in df.columns
                    if c not in table_spec.primary_key
                },
            )
            records = df.replace({np.nan: None}).to_dict(orient="records")
            conn.execute(insert_stmt, records)
            logger.info(
                f"Inserted {len(records)} rows into {table_spec.name} (PostgreSQL upsert path)"
            )
        elif insert_mode == "append":
            if engine.name == "postgresql":
                # Fast path using PostgreSQL's COPY FROM
                output = io.StringIO()
                df.to_csv(output, sep=",", header=False, index=False, na_rep="")
                output.seek(0)
                # Grab psycopg3 cursor to run the COPY command.
                with conn.connection.cursor() as cur:
                    with cur.copy(
                        f"""
                        COPY {table_spec.name} ({', '.join(df.columns)}) FROM STDIN
                        WITH (
                            FORMAT csv,
                            DELIMITER ',',
                            NULL '',
                            HEADER false
                        )"""
                    ) as copy:
                        copy.write(output.getvalue())
            else:
                # Standard path: use INSERT INTO measurements table.
                insert_stmt = sa.insert(table_spec.sa_table)
                # Ensure NaNs (for missing values from the CSV file) get inserted as NULLs.
                records = df.replace({np.nan: None}).to_dict(orient="records")
                conn.execute(insert_stmt, records)

        elif insert_mode == "insert_missing":
            # Use staging table to identify new (missing) rows.

            # Create staging table for bulk update
            staging_name = f"{table_spec.name}_staging_{str(uuid.uuid4())[:8]}"
            conn.execute(sa.text(f"DROP TABLE IF EXISTS {staging_name}"))
            conn.execute(
                sa.text(
                    f"""
                    CREATE TEMP TABLE {staging_name} AS
                    SELECT * FROM {table_spec.name} WHERE 0=1;
                    """
                )
            )
            staging_table = sa.Table(staging_name, sa.MetaData(), autoload_with=conn)

            # Bulk insert into staging table
            if engine.name == "postgresql":
                # Fast path using PostgreSQL's COPY FROM
                # https://www.postgresql.org/docs/current/sql-copy.html
                output = io.StringIO()
                df.to_csv(output, sep=",", header=False, index=False, na_rep="")
                output.seek(0)
                # Grab psycopg3 cursor to run the COPY command.
                with conn.connection.cursor() as cur:
                    with cur.copy(
                        f"""
                        COPY {staging_name} ({', '.join(df.columns)}) FROM STDIN
                        WITH (
                            FORMAT csv,
                            DELIMITER ',',
                            NULL '',
                            HEADER false
                        )"""
                    ) as copy:
                        copy.write(output.getvalue())
            else:
                # Standard path: INSERT INTO staging table.
                insert_stmt = sa.insert(staging_table)
                # Ensure NaNs (for missing values from the CSV file) get inserted as NULLs.
                records = df.replace({np.nan: None}).to_dict(orient="records")
                conn.execute(insert_stmt, records)

            logger.info(f"Inserted {len(df)} rows into staging table")

            # Merge into data table
            where_conditions = " AND ".join(
                [f"DataTable.{pk} = StagingTable.{pk}" for pk in table_spec.primary_key]
            )
            # Construct the final query
            insert_sql = f"""
                INSERT INTO {table_spec.name}
                SELECT StagingTable.*
                FROM {staging_table.name} AS StagingTable
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM {table_spec.name} AS DataTable
                    WHERE {where_conditions}
                )
                """

            c = conn.execute(sa.text(insert_sql))
            conn.execute(sa.text(f"DROP TABLE IF EXISTS {staging_name}"))
        else:
            raise AssertionError(f"Case not handled: mode={insert_mode}")

        duration = time.time() - start_time
        logger.info(
            "Inserted rows (mode=%s) into %s in %.2fs.",
            insert_mode,
            table_spec.name,
            duration,
        )

        # Update UpdateStatus in same transaction
        save_update_status(conn, update)


def save_update_status(conn: sa.Connection, s: UpdateStatus) -> None:
    if s.id is None:
        # INSERT new row
        conn.execute(
            sa.insert(sa_table_update_status).values(
                id=str(uuid.uuid4()),
                href=s.href,
                table_updated_time=s.table_updated_time.isoformat(),
                resource_updated_time=(
                    s.resource_updated_time.isoformat()
                    if s.resource_updated_time
                    else None
                ),
                etag=s.etag,
            )
        )
    else:
        # UPDATE existing row
        conn.execute(
            sa.update(sa_table_update_status)
            .where(sa_table_update_status.c.id == s.id)
            .values(
                table_updated_time=s.table_updated_time.isoformat(),
                resource_updated_time=(
                    s.resource_updated_time.isoformat()
                    if s.resource_updated_time
                    else None
                ),
                etag=s.etag,
            )
        )


def read_update_status(engine: sa.Engine) -> list[UpdateStatus]:
    """Reads all rows from the update_status table."""

    def parse_dt(s: str | None) -> datetime.datetime | None:
        return datetime.datetime.fromisoformat(s) if s else None

    statuses: list[UpdateStatus] = []

    stmt = sa.select(
        sa_table_update_status.c.id,
        sa_table_update_status.c.href,
        sa_table_update_status.c.resource_updated_time,
        sa_table_update_status.c.table_updated_time,
        sa_table_update_status.c.etag,
    )

    with engine.connect() as conn:
        result = conn.execute(stmt)
        for row in result.mappings():
            statuses.append(
                UpdateStatus(
                    id=row["id"],
                    href=row["href"],
                    resource_updated_time=parse_dt(row["resource_updated_time"]),
                    table_updated_time=parse_dt(row["table_updated_time"]),
                    etag=row["etag"],
                )
            )

    return statuses


def recreate_views(engine: sa.Engine) -> None:
    recreate_station_data_summary(engine)
    recreate_nearby_stations(engine)

    recreate_reference_period_stats_all(engine)
    recreate_reference_period_stats_month(engine)


def recreate_nearby_stations(
    engine: sa.Engine,
    max_distance_km: float = 80,
    max_neighbors=8,
    exclude_empty: bool = True,
) -> None:
    """Recreates the sa_table_x_nearby_stations table with nearby station info.

    Args:
        engine: the SQLAlchemy engine
        exclude_empty: If True, nearby stations are only included if they have
            measurement data.
    """
    with engine.begin() as conn:
        # Drop old table if exists
        sa_table_x_nearby_stations.drop(conn, checkfirst=True)
        # Create table
        sa_table_x_nearby_stations.create(conn)

    with engine.begin() as conn:
        stations = read_stations(conn, exclude_empty=exclude_empty)

    def _mkrec(s1: models.Station, s2: models.Station, dist_km: float):
        return {
            "from_station_abbr": s1.abbr,
            "from_station_name": s1.name,
            "from_station_canton": s1.canton,
            "to_station_abbr": s2.abbr,
            "to_station_name": s2.name,
            "to_station_canton": s2.canton,
            "distance_km": dist_km,
            "height_diff": s2.height_masl - s1.height_masl,
        }

    nearby_stations = [[] for _ in range(len(stations))]
    for i, j in itertools.combinations(range(len(stations)), 2):
        s1 = stations[i]
        s2 = stations[j]
        try:
            dist_km = geo.station_distance_meters(s1, s2, include_height=True) / 1000.0
        except ValueError:
            # Probably missing WGS or elevation data => skip
            continue
        if dist_km > max_distance_km:
            continue
        nearby_stations[i].append(_mkrec(s1, s2, dist_km))
        nearby_stations[j].append(_mkrec(s2, s1, dist_km))

    flat_records = []
    for records in nearby_stations:
        # Pick at most max_neighbors nearest nearby stations.
        records.sort(key=lambda s: (s["distance_km"], s["to_station_abbr"]))
        flat_records.extend(records[:max_neighbors])

    logger.info(
        "Determined %d nearby station pairs for %d stations",
        len(flat_records),
        len(stations),
    )
    if len(flat_records) == 0:
        return

    with engine.begin() as conn:
        insert = sa.insert(sa_table_x_nearby_stations)
        conn.execute(insert, flat_records)


def recreate_reference_period_stats_all(engine: sa.Engine) -> None:
    """(Re-)Creates a materialized view of summary data for the reference period 1991 to 2020."""
    with engine.begin() as conn:
        # Recreate table
        var_summary_stats_all.sa_table.drop(conn, checkfirst=True)
        var_summary_stats_all.sa_table.create(conn)

        # Insert data for aggregations.
        insert_summary_stats_from_daily_measurements(
            conn,
            stats_table=var_summary_stats_all,
            agg_name=AGG_NAME_REF_1991_2020,
            from_date=datetime.datetime(1991, 1, 1),
            to_date=datetime.datetime(2021, 1, 1),
        )


def recreate_reference_period_stats_month(engine: sa.Engine) -> None:
    with engine.begin() as conn:
        insert_summary_stats_from_daily_measurements(
            conn,
            stats_table=var_summary_stats_month,
            agg_name=AGG_NAME_REF_1991_2020,
            from_date=datetime.datetime(1991, 1, 1),
            to_date=datetime.datetime(2021, 1, 1),
        )


def insert_summary_stats_from_daily_measurements(
    conn: sa.Connection,
    stats_table: VarSummaryStatsTable,
    agg_name: str,
    from_date: datetime.datetime,
    to_date: datetime.datetime,
) -> None:
    """Inserts summary stats for the given reference period into sa_table_x_station_var_summary_stats."""
    # All daily measurement variables for which to build summary stats.
    daily_vars = [
        TEMP_DAILY_MIN,
        TEMP_DAILY_MAX,
        TEMP_DAILY_MEAN,
        PRECIP_DAILY_MM,
        SUNSHINE_DAILY_MINUTES,
        WIND_SPEED_DAILY_MEAN,
        GUST_PEAK_DAILY_MAX,
        ATM_PRESSURE_DAILY_MEAN,
    ]

    def _station_summary(
        df: pd.DataFrame, date_col: str, var_cols: list[str], granularity: str
    ) -> list[dict[str, Any]]:
        """Returns summary stats for all vars in var_cols as an SQL INSERT tuple."""
        params = []

        for (station_abbr, time_slice), grp in df.groupby(
            ["station_abbr", "time_slice"]
        ):
            non_null_cols = [v for v in var_cols if not grp[v].isna().all()]
            if not non_null_cols:
                continue  # No variables have any data.
            agg = grp[non_null_cols].agg(
                ["min", "max", "idxmin", "idxmax", "mean", "sum", "count"]
                + [pctl(10), pctl(25), pctl(50), pctl(75), pctl(90)]
            )
            for var in non_null_cols:
                a = agg[var]
                # Get reference_timestamp value at the index of the min/max value.
                min_date = grp.loc[a["idxmin"], date_col]
                max_date = grp.loc[a["idxmax"], date_col]
                params.append(
                    {
                        "agg_name": agg_name,
                        "station_abbr": station_abbr,
                        "variable": var,
                        "time_slice": time_slice,
                        "source_granularity": granularity,
                        "min_value": a["min"],
                        "min_value_date": min_date,
                        "mean_value": a["mean"],
                        "max_value": a["max"],
                        "max_value_date": max_date,
                        "p10_value": a["p10"],
                        "p10_value": a["p10"],
                        "p25_value": a["p25"],
                        "median_value": a["p50"],
                        "p75_value": a["p75"],
                        "p90_value": a["p90"],
                        "value_sum": a["sum"],
                        "value_count": int(a["count"]),
                    }
                )
        return params

    sql = f"""
        SELECT
            station_abbr,
            reference_timestamp,
            {', '.join(daily_vars)}
        FROM {TABLE_DAILY_MEASUREMENTS.name}
        WHERE reference_timestamp >= '{utc_datestr(from_date)}' 
            AND reference_timestamp < '{utc_datestr(to_date)}'
    """
    df = pd.read_sql_query(
        sql,
        conn,
        dtype={
            "station_abbr": str,
            "reference_timestamp": str,
            **{v: float for v in daily_vars},
        },
    )

    # Apply time_slicer to get time slice dimension.
    df["time_slice"] = stats_table.time_slicer(df)

    # Add DX_SOURCE_DATE_RANGE column containing epoch seconds.
    # This derived variable can be used to determine the overall time range
    # over which data was aggregated.
    delta_days = pd.to_datetime(df["reference_timestamp"]) - pd.Timestamp("1970-01-01")
    df[DX_SOURCE_DATE_RANGE] = delta_days / pd.Timedelta(days=1)

    # Summary stats across the whole period for all daily vars.
    params = _station_summary(
        df,
        date_col="reference_timestamp",
        var_cols=daily_vars + [DX_SOURCE_DATE_RANGE],
        granularity="daily",
    )

    def _day_count(series: pd.Series, condition: pd.Series):
        # Select condition where series had a value, else propagate NaN.
        return condition.where(series.notna()).astype(float)

    # Summary stats for generated annual metrics based on daily values.
    # For "day count" metrics, we calculate a true/false value per day
    # and interpret it as 1/0. Other metrics like Growing Degree Days
    # are derived from other daily variables and then summed up.
    # The solution respects NAs, i.e. if no data was available at all
    # for a given variable, it won't have summary stats.
    dc = pd.DataFrame(
        {
            "station_abbr": df["station_abbr"],
            "time_slice": df["time_slice"],
            "year": df["reference_timestamp"].str[:4] + "-01-01",
            # Day count metrics
            DX_SUMMER_DAYS_ANNUAL_COUNT: _day_count(
                df[TEMP_DAILY_MAX], df[TEMP_DAILY_MAX] >= 25
            ),
            DX_FROST_DAYS_ANNUAL_COUNT: _day_count(
                df[TEMP_DAILY_MIN], df[TEMP_DAILY_MIN] < 0
            ),
            DX_SUNNY_DAYS_ANNUAL_COUNT: _day_count(
                df[SUNSHINE_DAILY_MINUTES], df[SUNSHINE_DAILY_MINUTES] >= 6 * 60
            ),
            DX_TROPICAL_NIGHTS_ANNUAL_COUNT: _day_count(
                df[TEMP_DAILY_MIN], df[TEMP_DAILY_MIN] >= 20
            ),
            # Other derived metrics
            DX_GROWING_DEGREE_DAYS_ANNUAL_SUM: (
                0.5 * (df[TEMP_DAILY_MEAN].clip(upper=30) - 10).clip(lower=0)
            ),
        }
    )
    key_columns = ["station_abbr", "year", "time_slice"]
    # All variables defined in dc:
    dc_vars = [c for c in dc.columns if c not in key_columns]

    # Sum the 0/1 daily values to get day counts by year, retaining NaNs.
    def _nan_safe_sum(s):
        return s.sum() if s.notna().any() else float("nan")

    dcy = dc.groupby(key_columns)[dc_vars].agg(_nan_safe_sum).reset_index()
    # Now aggregate away the year (for each station) to compute summary stats
    # (same as above for plain daily variables).
    params.extend(
        _station_summary(dcy, date_col="year", var_cols=dc_vars, granularity="annual")
    )

    if params:
        insert_stmt = sa.insert(stats_table.sa_table)
        conn.execute(insert_stmt, params)


def recreate_station_data_summary(engine: sa.Engine) -> None:
    """Creates a materialized view of summary data per station_abbr.

    The summary stats in this table are calculated across the whole dataset.
    """

    with engine.begin() as conn:
        # Drop old table if exists
        sa_table_x_station_data_summary.drop(conn, checkfirst=True)

        # Create anew
        sa_table_x_station_data_summary.create(conn)

        # Populate
        conn.execute(
            sa.text(
                f"""
            INSERT INTO {sa_table_x_station_data_summary.name}
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

            FROM {sa_table_meta_stations.name} AS m
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
        )


def read_station(conn: sa.Connection, station_abbr: str) -> models.Station:
    """Returns data for the given station."""

    sql = sa.text(
        f"""
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
        FROM {sa_table_x_station_data_summary.name}
        WHERE station_abbr = :station_abbr
    """
    )

    result = conn.execute(sql, {"station_abbr": station_abbr}).mappings().first()
    if result is None:
        raise StationNotFoundError(f"No station found with abbr={station_abbr}")

    # Parse possible date strings into actual dates (None stays None)
    def d(v: str | None) -> datetime.date | None:
        return datetime.date.fromisoformat(v) if v else None

    return models.Station(
        abbr=result["station_abbr"],
        name=result["station_name"],
        canton=result["station_canton"],
        typ=result["station_type_en"],
        exposition=LocalizedString.from_nullable(
            de=result["station_exposition_de"],
            fr=result["station_exposition_fr"],
            it=result["station_exposition_it"],
            en=result["station_exposition_en"],
        ),
        url=LocalizedString.from_nullable(
            de=result["station_url_de"],
            fr=result["station_url_fr"],
            it=result["station_url_it"],
            en=result["station_url_en"],
        ),
        height_masl=result["station_height_masl"],
        coordinates_wgs84_lat=result["station_coordinates_wgs84_lat"],
        coordinates_wgs84_lon=result["station_coordinates_wgs84_lon"],
        temperature_min_date=d(result["tre200d0_min_date"]),
        temperature_max_date=d(result["tre200d0_max_date"]),
        precipitation_min_date=d(result["rre150d0_min_date"]),
        precipitation_max_date=d(result["rre150d0_max_date"]),
    )


def read_stations(
    conn: sa.Connection,
    cantons: list[str] | None = None,
    exclude_empty: bool = True,
) -> list[models.Station]:
    """Returns all stations matching the given criteria."""

    base_sql = f"""
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
        FROM {sa_table_x_station_data_summary.name}
    """

    filters = []
    params: dict[str, Any] = {}
    bindparams = []

    # Canton filter using tuple bind
    if cantons:
        filters.append("station_canton IN :cantons")
        params["cantons"] = tuple(cantons)
        # Let SQLAlchemy expand 'cantons'
        bindparams.append(sa.bindparam("cantons", expanding=True))

    # Exclude stations with no data
    if exclude_empty:
        filters.append("(tre200d0_count > 0 AND rre150d0_count > 0)")

    # Combine filters
    sql = base_sql
    if filters:
        sql += " WHERE " + " AND ".join(filters)
    sql += " ORDER BY station_name"

    sa_sql = sa.text(sql)
    if bindparams:
        sa_sql = sa_sql.bindparams(*bindparams)
    result = conn.execute(sa_sql, params).mappings().all()

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
        for row in result
    ]


def read_nearby_stations(
    conn: sa.Connection,
    station_abbr: str,
) -> list[models.NearbyStation]:
    sql = f"""
        SELECT
            to_station_abbr AS abbr,
            to_station_name AS name,
            to_station_canton AS canton,
            distance_km,
            height_diff
        FROM {sa_table_x_nearby_stations.name}
        WHERE from_station_abbr = :station_abbr
        ORDER BY distance_km, to_station_abbr
    """

    result = conn.execute(sa.text(sql), {"station_abbr": station_abbr}).mappings().all()

    return [
        models.NearbyStation(
            abbr=row["abbr"],
            name=row["name"],
            canton=row["canton"],
            distance_km=row["distance_km"],
            height_diff=row["height_diff"],
        )
        for row in result
    ]


def _sql_filter_by_period(period: str) -> str:
    """Returns an SQL expression that matches reference_timestamp values for the given period.

    NOTE: This function relies on the fact that `reference_timestamp` is
        stored as a string that always starts with YYYY-MM.
    """
    if period == "all":
        return "1=1"

    if period.isdigit():
        return f"SUBSTR(reference_timestamp, 6, 2) = '{int(period):02d}'"

    if period == "spring":
        return "SUBSTR(reference_timestamp, 6, 2) IN ('03', '04', '05')"
    elif period == "summer":
        return "SUBSTR(reference_timestamp, 6, 2) IN ('06', '07', '08')"
    elif period == "autumn":
        return "SUBSTR(reference_timestamp, 6, 2) IN ('09', '10', '11')"
    elif period == "winter":
        return "SUBSTR(reference_timestamp, 6, 2) IN ('12', '01', '02')"

    raise ValueError(f"Invalid period {period} for SQL filter")


def read_daily_measurements(
    conn: sa.Connection,
    station_abbr: str,
    columns: list[str] | None = None,
    period: str | None = None,
    from_year: int | None = None,
    to_year: int | None = None,
) -> pd.DataFrame:
    """Returns daily measurements matching the provided filters.

    The returned DataFrame has a datetime (reference_timestamp) index.
    """
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
    filters = ["station_abbr = :station_abbr"]
    params = {"station_abbr": station_abbr}

    if period is not None:
        filters.append(_sql_filter_by_period(period))

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
        sa.text(sql),
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


def utc_datestr(d: datetime.datetime) -> str:
    """Returns the given datetime as a UTC date string in ISO format.

    Example: "2025-03-31"

    If d is a naive datetime (no tzinfo), it is assumed to be in UTC.
    """
    if d.tzinfo is not None:
        # datetime.UTC is only available since Python 3.11
        d = d.astimezone(datetime.timezone.utc)
    return d.strftime("%Y-%m-%d")


def read_hourly_measurements(
    conn: sa.Connection,
    station_abbr: str,
    from_date: datetime.datetime,
    to_date: datetime.datetime,
    columns: list[str] | None = None,
    limit: int = -1,
) -> pd.DataFrame:
    if columns is None:
        columns = [TEMP_HOURLY_MIN, TEMP_HOURLY_MEAN, TEMP_HOURLY_MAX, PRECIP_HOURLY_MM]

    # Validate column names to prevent SQL injection
    if not all(re.search(r"^[a-z][a-zA-Z0-9_]*$", c) for c in columns):
        raise ValueError(f"Invalid columns: {','.join(columns)}")

    select_columns = ["station_abbr", "reference_timestamp"] + columns
    sql = f"""
        SELECT {', '.join(select_columns)}
        FROM {TABLE_HOURLY_MEASUREMENTS.name}
    """
    # Filter by station.
    filters = ["station_abbr = :station_abbr"]
    params = {"station_abbr": station_abbr}

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
    if limit > 0:
        sql += f" LIMIT {limit}"

    return pd.read_sql_query(
        sa.text(sql),
        conn,
        params=params,
        parse_dates=["reference_timestamp"],
        index_col="reference_timestamp",
    )


def read_monthly_measurements(
    conn: sa.Connection,
    station_abbr: str,
    from_date: datetime.datetime,
    to_date: datetime.datetime,
    columns: list[str] | None = None,
    limit: int = -1,
) -> pd.DataFrame:
    """Reads rows from the monthly measurements table.

    Args:
        from_date: inclusive lower bound of the time range to read.
        to_date: exclusive upper bound of the time range to read.
    """
    if columns is None:
        columns = [
            TEMP_MONTHLY_MIN,
            TEMP_MONTHLY_MEAN,
            TEMP_MONTHLY_MAX,
            PRECIP_MONTHLY_MM,
        ]

    # Validate column names to prevent SQL injection
    if not all(re.search(r"^[a-z][a-zA-Z0-9_]*$", c) for c in columns):
        raise ValueError(f"Invalid columns: {','.join(columns)}")

    select_columns = ["station_abbr", "reference_timestamp"] + columns
    sql = f"""
        SELECT {', '.join(select_columns)}
        FROM {TABLE_MONTHLY_MEASUREMENTS.name}
    """
    # Filter by station.
    filters = ["station_abbr = :station_abbr"]
    params = {"station_abbr": station_abbr}

    if from_date is not None:
        filters.append(f"reference_timestamp >= '{utc_datestr(from_date)}'")
    if to_date is not None:
        filters.append(f"reference_timestamp < '{utc_datestr(to_date)}'")

    # Filter out any row that has only NULL measurements.
    if columns:
        non_null = " OR ".join(f"{c} IS NOT NULL" for c in columns)
        filters.append(f"({non_null})")

    sql += " WHERE " + " AND ".join(filters)
    sql += " ORDER BY reference_timestamp ASC"
    if limit > 0:
        sql += f" LIMIT {limit}"

    return pd.read_sql_query(
        sa.text(sql),
        conn,
        params=params,
        parse_dates=["reference_timestamp"],
        index_col="reference_timestamp",
    )


def _column_to_dtype(col: sa.Column) -> Any:
    if isinstance(col.type, sa.Float):
        return float
    if isinstance(col.type, sa.String):
        return str
    if isinstance(col.type, sa.Integer):
        return int
    raise ValueError(f"Cannot determine dtype for {col.name} of type {col.type}")


def read_var_summary_stats_all(
    conn: sa.Connection,
    agg_name: str,
    station_abbr: str | None = None,
    variables: Collection[str] | None = None,
) -> pd.DataFrame:
    """Convenience wrapper for read_station_var_summary_stats_generic discarding the time_slice dimension."""
    df = read_summary_stats(
        conn,
        table=var_summary_stats_all.sa_table,
        agg_name=agg_name,
        station_abbr=station_abbr,
        variables=variables,
    )
    if df.empty:
        return df
    # Drop the constant '*' time_slice dimension.
    return df.xs(TS_ALL, level="time_slice")


def read_var_summary_stats_month(
    conn: sa.Connection,
    agg_name: str,
    station_abbr: str | None = None,
    months: list[int] | None = None,
    variables: Collection[str] | None = None,
) -> pd.DataFrame:
    return read_summary_stats(
        conn,
        table=var_summary_stats_month.sa_table,
        agg_name=agg_name,
        time_slices=[ts_month(m) for m in months] if months else None,
        station_abbr=station_abbr,
        variables=variables,
    )


def read_summary_stats(
    conn: sa.Connection,
    table: sa.Table,
    agg_name: str,
    time_slices: list[str] | None = None,
    station_abbr: str | None = None,
    variables: Collection[str] | None = None,
) -> pd.DataFrame:
    """
    Reads summary stats for variables for the aggregation with the given `agg_name`.

    Usage examples:

        # Get summary values of variable rre150dn for station BER and the TS_ALL time slice:
        df = read_station_var_summary_stats_generic(
            conn,
            sa_table_x_station_var_summary_stats,
            AGG_NAME_REF_1991_2020,
            "BER"
        )
        p = df.loc[("BER", "rre150dn", TS_ALL)]
        print("Min precipitation level was", p["min_value"], "on", p["min_value_date"])

    Args:
        conn: An active SQLAlchemy database connection.
        table: The table to read from.
        agg_name: The name of the aggregataion to read data for.
        time_slices: Time slices to read. Returns all time slices if None.
        station_abbr: Optional abbreviation of the station to filter by.
        variables: Optional collection of variable names to filter by.

    Returns:
        A pandas DataFrame with a 3-layer MultiIndex:
        ['station_abbr', 'variable', 'time_slice']
        and columns the summary statistics (p10, p25, mean, etc.).

        Returns a generic empty DataFrame if no data matches the filters.
    """
    filters = ["agg_name = :agg_name"]
    params = {"agg_name": agg_name}
    if station_abbr:
        filters.append("station_abbr = :station_abbr")
        params["station_abbr"] = station_abbr
    if variables:
        placeholders = []
        for i, var in enumerate(variables):
            p = f"var{i}"
            placeholders.append(f":{p}")
            params[p] = var
        filters.append(f"variable IN ({', '.join(placeholders)})")
    if time_slices:
        placeholders = []
        for i, var in enumerate(time_slices):
            p = f"ts{i}"
            placeholders.append(f":{p}")
            params[p] = var
        filters.append(f"time_slice IN ({', '.join(placeholders)})")

    excluded_col_names = set(["agg_name"])
    columns = [c for c in table.columns if c.name not in excluded_col_names]
    sql = f"""
        SELECT
            {', '.join(c.name for c in columns)}
        FROM {table.name}
        """
    if filters:
        sql += f"WHERE {' AND '.join(filters)}"

    df_long = pd.read_sql_query(
        sa.text(sql),
        conn,
        params=params,
        dtype={c.name: _column_to_dtype(c) for c in columns},
    )
    if df_long.empty:
        return pd.DataFrame()

    return df_long.set_index(["station_abbr", "variable", "time_slice"])


def table_stats(engine: sa.Engine, user: str) -> list[models.DBTableStats]:
    with engine.begin() as conn:
        sql = sql_queries.psql_total_bytes(user)
        result = conn.execute(sql).mappings().all()
        return [
            models.DBTableStats(
                schema_name=r["schema"],
                table=r["table"],
                total_size=r["total_size"],
                total_bytes=r["total_bytes"],
            )
            for r in result
        ]
