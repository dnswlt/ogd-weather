import datetime
import logging
import os
from typing import Any, Collection, Iterable
import uuid
import pandas as pd
from pydantic import BaseModel
import re
import sqlalchemy as sa

from . import models
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

        # Measurement columns are always REAL → Float in SQLAlchemy
        measurement_columns = [
            sa.Column(col_name, sa.Float) for col_name in self.measurements
        ]

        return sa.Table(self.name, metadata, *(pk_columns + measurement_columns))


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
}

# Derived metric names (prefix DX_):
DX_SUNNY_DAYS_ANNUAL_COUNT = "sunny_days_annual_count"
DX_SUMMER_DAYS_ANNUAL_COUNT = "summer_days_annual_count"
DX_FROST_DAYS_ANNUAL_COUNT = "frost_days_annual_count"
DX_TROPICAL_NIGHTS_ANNUAL_COUNT = "tropical_nights_annual_count"

DX_GROWING_DEGREE_DAYS_ANNUAL_SUM = "growing_degree_days_annual_sum"

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
        WIND_SPEED_DAILY_MEAN,
        WIND_DIRECTION_DAILY_MEAN,
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
    sa.Column("station_height_masl", sa.Float),
    sa.Column("station_height_barometer_masl", sa.Float),
    sa.Column("station_coordinates_lv95_east", sa.Float),
    sa.Column("station_coordinates_lv95_north", sa.Float),
    sa.Column("station_coordinates_wgs84_lat", sa.Float),
    sa.Column("station_coordinates_wgs84_lon", sa.Float),
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
    sa.Column("station_height_masl", sa.Float),
    sa.Column("station_coordinates_wgs84_lat", sa.Float),
    sa.Column("station_coordinates_wgs84_lon", sa.Float),
    sa.Column("tre200d0_count", sa.Integer, nullable=False),
    sa.Column("tre200d0_min_date", sa.Text),
    sa.Column("tre200d0_max_date", sa.Text),
    sa.Column("rre150d0_count", sa.Integer, nullable=False),
    sa.Column("rre150d0_min_date", sa.Text),
    sa.Column("rre150d0_max_date", sa.Text),
)


sa_table_x_station_var_summary_stats = sa.Table(
    "x_station_var_summary_stats",
    metadata,
    sa.Column("agg_name", sa.Text, primary_key=True),
    sa.Column("station_abbr", sa.Text, primary_key=True),
    sa.Column("variable", sa.Text, primary_key=True),
    sa.Column("source_granularity", sa.Text),
    sa.Column("min_value", sa.Float),
    sa.Column("min_value_date", sa.Text),
    sa.Column("mean_value", sa.Float),
    sa.Column("max_value", sa.Float),
    sa.Column("max_value_date", sa.Text),
    sa.Column("value_sum", sa.Float),
    sa.Column("value_count", sa.Integer),
)


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


def insert_csv_metadata(
    engine: sa.Engine,
    table: sa.Table,
    csv_file: str,
    sep: str = ";",
    encoding: str = "cp1252",
) -> None:
    """Loads CSV and insert rows into an existing metadata table.

    - Requires all table columns to be present in the CSV.
    """

    # Columns defined in the SQLAlchemy table (in order)
    table_columns = [col.name for col in table.columns]

    # Load CSV
    df = pd.read_csv(csv_file, sep=sep, encoding=encoding)

    # Validate all columns present
    missing = [c for c in table_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_file} is missing required columns: {missing}")

    # Select only required columns, in correct order
    df = df[table_columns]

    # Prepare insert
    placeholders = ", ".join([f":{c}" for c in table_columns])
    insert_sql = sa.text(
        f"INSERT OR IGNORE INTO {table.name} ({', '.join(table_columns)}) VALUES ({placeholders})"
    )

    # Convert to list-of-dicts (SQLAlchemy will bind params safely)
    rows = df.to_dict(orient="records")

    # Insert in one transaction
    with engine.begin() as conn:
        conn.execute(insert_sql, rows)


def insert_csv_data(
    base_dir: str,
    engine: sa.Engine,
    table_spec: DataTableSpec,
    csv_filenames: list[str] | None = None,
) -> None:
    """Inserts data from CSV files.

    Column names in the DB tables are identical to column names in the CSV (e.g.,
    station_abbr, reference_timestamp, tre200d0).

    Dates are stored as TEXT in ISO format "YYYY-MM-DD HH:MM:SSZ" and use UTC time.

    The time part is included only for sub-daily measurements, otherwise "YYYY-MM-DD" is used.
    """

    files = [os.path.join(base_dir, f) for f in csv_filenames or []]

    csv_dtype = {}
    for c in table_spec.measurements:
        csv_dtype[c] = float

    columns = table_spec.primary_key + table_spec.measurements
    for fname in files:
        logger.info(f"Importing {fname} into database...")
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

        # Insert rows, ignoring duplicates
        with engine.begin() as conn:
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
            # Bulk insert into staging table
            staging_table = sa.Table(staging_name, sa.MetaData(), autoload_with=conn)
            insert_stmt = sa.insert(staging_table)
            records = df.to_dict(orient="records")
            conn.execute(insert_stmt, records)
            logger.info(f"Inserted {len(records)} rows into staging table")
            # Merge into data table
            merge_sql = sa.text(
                f"""
                INSERT INTO {table_spec.name}
                SELECT StagingTable.*
                FROM {staging_table.name} AS StagingTable
                LEFT JOIN {table_spec.name} AS DataTable USING ({', '.join(table_spec.primary_key)})
                WHERE DataTable.{table_spec.primary_key[0]} IS NULL
                """
            )
            conn.execute(merge_sql)
            conn.execute(sa.text(f"DROP TABLE IF EXISTS {staging_name}"))


def save_update_status(engine: sa.Engine, statuses: Iterable[UpdateStatus]) -> None:
    """Inserts or updates the statuses in the update status table."""
    with engine.begin() as conn:
        for s in statuses:
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
    recreate_station_var_summary_stats(engine)


def recreate_station_var_summary_stats(engine: sa.Engine) -> None:
    """(Re-)Creates a materialized view of summary data for the reference period 1991 to 2020."""
    with engine.begin() as conn:
        # Recreate table
        sa_table_x_station_var_summary_stats.drop(conn, checkfirst=True)
        sa_table_x_station_var_summary_stats.create(conn)
        # Insert data for aggregations (for now, just one).
        insert_ref_1991_2020_summary_stats(conn)


def insert_ref_1991_2020_summary_stats(conn: sa.Connection) -> None:
    """Inserts the 1991-2020 reference period into sa_table_x_station_var_summary_stats."""
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
        for station_abbr, grp in df.groupby("station_abbr"):
            for var in var_cols:
                if grp[var].isna().all():
                    continue  # No data for this variable
                a = grp[var].agg(
                    ["min", "max", "idxmin", "idxmax", "mean", "sum", "count"]
                )
                # Get reference_timestamp value at the index of the min/max value.
                # Convert to Python datetime, which conn.execute understands.
                min_date = grp.loc[a["idxmin"], date_col]
                max_date = grp.loc[a["idxmax"], date_col]
                params.append(
                    {
                        "agg_name": AGG_NAME_REF_1991_2020,
                        "station_abbr": station_abbr,
                        "variable": var,
                        "source_granularity": granularity,
                        "min_value": a["min"],
                        "min_value_date": min_date,
                        "mean_value": a["mean"],
                        "max_value": a["max"],
                        "max_value_date": max_date,
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
        WHERE reference_timestamp >= '1991-01-01' AND reference_timestamp < '2021-01-01'
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

    # Summary stats across the whole period for all daily vars.
    params = _station_summary(
        df, date_col="reference_timestamp", var_cols=daily_vars, granularity="daily"
    )

    def _day_count(series, condition):
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
    # All variables defined in dc:
    dc_vars = [c for c in dc.columns if c not in ("station_abbr", "year")]

    # Sum the 0/1 daily values to get day counts by year, retaining NaNs.
    def _nan_safe_sum(s):
        return s.sum() if s.notna().any() else float("nan")

    dcy = dc.groupby(["station_abbr", "year"])[dc_vars].agg(_nan_safe_sum).reset_index()
    # Now aggregate away the year (for each station) to compute summary stats
    # (same as above for plain daily variables).
    params.extend(
        _station_summary(dcy, date_col="year", var_cols=dc_vars, granularity="annual")
    )

    if params:
        insert_stmt = sa.insert(sa_table_x_station_var_summary_stats)
        conn.execute(insert_stmt, params)

    conn.commit()


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


def read_daily_measurements(
    conn: sa.Connection,
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
    filters = ["station_abbr = :station_abbr"]
    params = {"station_abbr": station_abbr}

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
    conn: sa.Connection,
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

    return pd.read_sql_query(
        sql,
        conn,
        params=params,
        parse_dates=["reference_timestamp"],
        index_col="reference_timestamp",
    )


def read_station_var_summary_stats(
    conn: sa.Connection,
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
        conn: An active SQLAlchemy database connection.
        agg_name: The name of the aggregataion to read data for.
        station_abbr: Optional abbreviation of the station to filter by.
        variables: Optional collection of variable names to filter by.

    Returns:
        A pandas DataFrame in wide format (rows indexed by 'station_abbr',
        columns as (variable, measurement_type)).
        Returns an empty DataFrame if no data matches the filters.
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

    measure_cols = [
        "source_granularity",
        "min_value",
        "min_value_date",
        "mean_value",
        "max_value",
        "max_value_date",
        "value_sum",
        "value_count",
    ]
    sql = f"""
        SELECT
            station_abbr,
            variable,
            {', '.join(measure_cols)}
        FROM {sa_table_x_station_var_summary_stats.name}
    """
    if filters:
        sql += f"WHERE {' AND '.join(filters)}"

    df_long = pd.read_sql_query(
        sql,
        conn,
        params=params,
        dtype={
            "station_abbr": str,
            "variable": str,
            "source_granularity": str,
            "min_value": float,
            "min_value_date": str,
            "mean_value": float,
            "max_value": float,
            "max_value_date": str,
            "value_sum": float,
            "value_count": float,
        },
    )
    if df_long.empty:
        return pd.DataFrame(index=pd.Index([], name="station_abbr"), dtype=str)

    return df_long.set_index(["station_abbr", "variable"]).unstack().swaplevel(axis=1)
