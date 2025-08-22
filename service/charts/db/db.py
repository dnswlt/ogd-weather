"""Contains functions to query and manipulate the database."""

import datetime
import io
import itertools
import logging
import os
import time
from typing import Any, Collection
from urllib.parse import urlparse
import uuid
import numpy as np
import pandas as pd
from pydantic import BaseModel
import re
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from service.charts import geo
from service.charts import models
from service.charts.base.errors import StationNotFoundError
from service.charts.models import LocalizedString

from . import sql_queries
from . import constants as dc
from . import schema as ds


logger = logging.getLogger("db")


class UpdateStatus(BaseModel):
    """Represents a row in the update_status tracking table."""

    id: str | None  # UUID-4, empty if this is a new record
    href: str
    table_updated_time: datetime.datetime
    resource_updated_time: datetime.datetime | None = None
    etag: str | None = None

    _is_not_modified: bool = False

    def filename(self):
        """Returns the filename part of href."""
        return os.path.basename(urlparse(self.href).path)

    def mark_not_modified(
        self,
        table_updated_time: datetime.datetime,
        resource_updated_time: datetime.datetime,
    ):
        """Marks the status as "Not Modified".

        This is used to avoid repeatedly checking a resource for updates
        by remembering the last time we checked.
        """
        self.table_updated_time = table_updated_time
        self.resource_updated_time = resource_updated_time
        self._is_not_modified = True

    def is_not_modified(self):
        return self._is_not_modified

    def update(
        self,
        table_updated_time: datetime.datetime,
        resource_updated_time: datetime.datetime,
        etag: str | None,
    ):
        """Updates this status according to the new timestamps and Etag."""
        self.table_updated_time = table_updated_time
        self.resource_updated_time = resource_updated_time
        self.etag = etag


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
    table_spec: ds.DataTableSpec,
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
    logger.info(f"Importing {filename} into database")
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
            sa.insert(ds.sa_table_update_status).values(
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
            sa.update(ds.sa_table_update_status)
            .where(ds.sa_table_update_status.c.id == s.id)
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
        ds.sa_table_update_status.c.id,
        ds.sa_table_update_status.c.href,
        ds.sa_table_update_status.c.resource_updated_time,
        ds.sa_table_update_status.c.table_updated_time,
        ds.sa_table_update_status.c.etag,
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
    recreate_station_var_availability(engine)
    recreate_nearby_stations(engine)
    recreate_climate_normals_stats_tables(engine)


def recreate_climate_normals_stats_tables(engine: sa.Engine):
    logger.info("Reading daily measurements (1991-2020) from DB")
    daily_measurements = _read_all_daily_measurements(
        engine, datetime.datetime(1991, 1, 1), datetime.datetime(2021, 1, 1)
    )
    logger.info("Recreating reference period (1991-2020) 'all' stats")
    recreate_reference_period_stats(
        engine, ds.var_summary_stats_all, daily_measurements
    )
    logger.info("Recreating reference period (1991-2020) 'month' stats")
    recreate_reference_period_stats(
        engine, ds.var_summary_stats_month, daily_measurements
    )


def recreate_nearby_stations(
    engine: sa.Engine,
    max_distance_km: float = 80,
    max_stations=8,
    exclude_empty: bool = True,
) -> None:
    """Recreates the ds.sa_table_x_nearby_stations table with nearby station info.

    Args:
        engine: the SQLAlchemy engine
        exclude_empty: If True, nearby stations are only included if they have
            measurement data.
        max_distance_km: maximum distance in km to consider a station "nearby".
        max_stations: maximum number of nearby stations to include in the result
            (nearest first).
    """

    logger.info("Recreating station data summary")

    with engine.begin() as conn:
        # Drop old table if exists
        ds.sa_table_x_nearby_stations.drop(conn, checkfirst=True)
        # Create table
        ds.sa_table_x_nearby_stations.create(conn)

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
        flat_records.extend(records[:max_stations])

    logger.info(
        "Determined %d nearby station pairs for %d stations",
        len(flat_records),
        len(stations),
    )
    if len(flat_records) == 0:
        return

    with engine.begin() as conn:
        insert = sa.insert(ds.sa_table_x_nearby_stations)
        conn.execute(insert, flat_records)


def recreate_reference_period_stats(
    engine: sa.Engine,
    stats_table: ds.VarSummaryStatsTable,
    daily_measurements: pd.DataFrame | None = None,
) -> None:
    """(Re-)Creates a materialized view of summary data for the reference period 1991 to 2020.

    Args:
        engine: The SQLAlchemy engine to use.
        stats_table: the summary stats table to write to.
    """
    with engine.begin() as conn:
        # Recreate table
        stats_table.sa_table.drop(conn, checkfirst=True)
        stats_table.sa_table.create(conn)

        # Insert data for aggregations.
        insert_summary_stats_from_daily_measurements(
            conn,
            stats_table=stats_table,
            agg_name=dc.AGG_NAME_REF_1991_2020,
            from_date=datetime.datetime(1991, 1, 1),
            to_date=datetime.datetime(2021, 1, 1),
            daily_measurements=daily_measurements,
        )


def _var_summary_stats(
    df: pd.DataFrame,
    agg_name: str,
    date_col: str,
    var_cols: list[str],
    granularity: str,
) -> list[dict[str, Any]]:
    """Returns summary stats for all vars in var_cols as a list of SQL INSERT records."""

    # This code is optimized for speed, so only change it if you know what you're doing.
    id_cols = [date_col, "station_abbr", "time_slice"]
    # Convert to long format for efficient groupby. Drop rows with NaN values.
    long = (
        df[id_cols + var_cols]
        .melt(id_vars=id_cols, var_name="variable", value_name="value")
        .dropna(subset="value")
    )
    # Aggregate away the date and compute statistics.
    # Compute "standard" and quantile stats separately for maximum efficiency
    # (avoid calling a Python UDF for each quantile separately), then join.
    grp = long.groupby(
        ["station_abbr", "time_slice", "variable"], sort=False, observed=True
    )
    std = grp["value"].agg(["min", "max", "idxmin", "idxmax", "mean", "sum", "count"])
    qs = grp["value"].quantile(q=[0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    res = pd.concat([std, qs], axis=1)

    # Ensure field names match SQL column names.
    records = res.reset_index().rename(
        columns={
            "min": "min_value",
            "mean": "mean_value",
            "max": "max_value",
            0.1: "p10_value",
            0.25: "p25_value",
            0.5: "median_value",
            0.75: "p75_value",
            0.9: "p90_value",
            "sum": "value_sum",
            "count": "value_count",
        }
    )
    # Look up min/max dates in the original long DataFrame.
    records["min_value_date"] = records["idxmin"].map(long[date_col])
    records["max_value_date"] = records["idxmax"].map(long[date_col])
    # Constant fields.
    records["agg_name"] = agg_name
    records["source_granularity"] = granularity

    return records.to_dict(orient="records")


def _read_all_daily_measurements(
    engine: sa.Engine,
    from_date: datetime.datetime,
    to_date: datetime.datetime,
):
    """Reads ALL ROWS from ds.TABLE_DAILY_MEASUREMENTS in the range [from_date, to_date) into memory."""
    daily_vars = ds.TABLE_DAILY_MEASUREMENTS.measurements
    sql = f"""
        SELECT
            station_abbr,
            reference_timestamp,
            {', '.join(daily_vars)}
        FROM {ds.TABLE_DAILY_MEASUREMENTS.name}
        WHERE reference_timestamp >= '{utc_datestr(from_date)}' 
            AND reference_timestamp < '{utc_datestr(to_date)}'
    """
    with engine.begin() as conn:
        df = pd.read_sql_query(
            sql,
            conn,
            dtype={
                "station_abbr": str,
                "reference_timestamp": str,
                **{v: np.float32 for v in daily_vars},
            },
        )
    logger.info(
        "Read %d rows from %s (%.1f MiB)",
        len(df),
        ds.TABLE_DAILY_MEASUREMENTS.name,
        df.memory_usage().sum() / (1 << 20),
    )
    return df


def insert_summary_stats_from_daily_measurements(
    conn: sa.Connection,
    stats_table: ds.VarSummaryStatsTable,
    agg_name: str,
    from_date: datetime.datetime,
    to_date: datetime.datetime,
    daily_measurements: pd.DataFrame | None = None,
) -> None:
    """Inserts summary stats for the given reference period into the given stats_table."""
    # All daily measurement variables for which to build summary stats.
    daily_vars = ds.TABLE_DAILY_MEASUREMENTS.measurements

    if daily_measurements is None:
        df = _read_all_daily_measurements(conn.engine, from_date, to_date)
    else:
        df = daily_measurements.copy()

    # Apply time_slicer to get time slice dimension.
    df["time_slice"] = stats_table.time_slicer(df)

    # Add DX_SOURCE_DATE_RANGE column containing epoch seconds.
    # This derived variable can be used to determine the overall time range
    # over which data was aggregated.
    delta_days = pd.to_datetime(df["reference_timestamp"]) - pd.Timestamp("1970-01-01")
    df[dc.DX_SOURCE_DATE_RANGE] = delta_days / pd.Timedelta(days=1)

    # Summary stats across the whole period for all daily vars.
    params = _var_summary_stats(
        df,
        agg_name=agg_name,
        date_col="reference_timestamp",
        var_cols=daily_vars + [dc.DX_SOURCE_DATE_RANGE],
        granularity="daily",
    )

    def _day_count(series: pd.Series, condition: pd.Series):
        # Select condition where series had a value, else propagate NaN.
        return condition.where(series.notna()).astype(float)

    # Summary stats for generated summary stats based on daily values.
    # For "day count" metrics, we calculate a true/false value per day
    # and interpret it as 1/0. Other metrics like Growing Degree Days
    # are derived from other daily variables and then summed up.
    # The solution respects NAs, i.e. if no data was available at all
    # for a given variable, it won't have summary stats.
    df_gen = pd.DataFrame(
        {
            "station_abbr": df["station_abbr"],
            "time_slice": df["time_slice"],
            "year": df["reference_timestamp"].str[:4] + "-01-01",
            # Day count metrics
            dc.DX_SUMMER_DAYS_ANNUAL_COUNT: _day_count(
                df[dc.TEMP_DAILY_MAX], df[dc.TEMP_DAILY_MAX] >= 25
            ),
            dc.DX_FROST_DAYS_ANNUAL_COUNT: _day_count(
                df[dc.TEMP_DAILY_MIN], df[dc.TEMP_DAILY_MIN] < 0
            ),
            dc.DX_RAIN_DAYS_ANNUAL_COUNT: _day_count(
                df[dc.PRECIP_DAILY_MM], df[dc.PRECIP_DAILY_MM] >= 1.0
            ),
            dc.DX_SUNNY_DAYS_ANNUAL_COUNT: _day_count(
                df[dc.SUNSHINE_DAILY_MINUTES], df[dc.SUNSHINE_DAILY_MINUTES] >= 6 * 60
            ),
            dc.DX_TROPICAL_NIGHTS_ANNUAL_COUNT: _day_count(
                df[dc.TEMP_DAILY_MIN], df[dc.TEMP_DAILY_MIN] >= 20
            ),
            # Other derived metrics
            dc.DX_GROWING_DEGREE_DAYS_ANNUAL_SUM: (
                0.5 * (df[dc.TEMP_DAILY_MEAN].clip(upper=30) - 10).clip(lower=0)
            ),
            dc.DX_PRECIP_TOTAL: df[dc.PRECIP_DAILY_MM],
        }
    )
    key_columns = ["station_abbr", "year", "time_slice"]
    # All variables defined in df_gen:
    df_gen_vars = [c for c in df_gen.columns if c not in key_columns]

    # Sum the 0/1 daily values to get day counts by year, retaining NaNs.
    df_y = df_gen.groupby(key_columns)[df_gen_vars].sum(min_count=1).reset_index()
    # Now aggregate away the year (for each station) to compute summary stats
    # (same as above for plain daily variables).
    params.extend(
        _var_summary_stats(
            df_y,
            agg_name=agg_name,
            date_col="year",
            var_cols=df_gen_vars,
            granularity="annual",
        )
    )

    if params:
        logger.info("Inserting %d rows into %s", len(params), stats_table.sa_table.name)
        insert_stmt = sa.insert(stats_table.sa_table)
        conn.execute(insert_stmt, params)


def recreate_station_var_availability(engine: sa.Engine) -> None:
    """Recreates the ds.sa_table_x_station_var_availability materialized view.

    Computes availability information for each variable (measurement) in
    ds.TABLE_DAILY_MEASUREMENTS and stores it in long format ("unpivot").
    """
    logger.info("Recreating station variable availability")

    with engine.begin() as conn:
        ds.sa_table_x_station_var_availability.drop(conn, checkfirst=True)
        ds.sa_table_x_station_var_availability.create(conn)

    with engine.begin() as conn:
        select_agg_avail = ", ".join(
            f"""
                SUM(CASE WHEN {m} IS NOT NULL THEN 1 ELSE 0 END) AS {m}_count,
                MIN(CASE WHEN {m} IS NOT NULL THEN reference_timestamp END) AS {m}_min_date,
                MAX(CASE WHEN {m} IS NOT NULL THEN reference_timestamp END) AS {m}_max_date
            """
            for m in ds.TABLE_DAILY_MEASUREMENTS.measurements
        )
        # To unpivot, we UNION ALL the aggregation results for each variable.
        select_union_all = "\nUNION ALL\n".join(
            f"SELECT station_abbr, '{m}', {m}_count, {m}_min_date, {m}_max_date FROM Aggregated"
            for m in ds.TABLE_DAILY_MEASUREMENTS.measurements
        )

        insert_sql = sa.text(
            f"""
                WITH Aggregated AS (
                    SELECT 
                        station_abbr,
                        {select_agg_avail}
                    FROM {ds.TABLE_DAILY_MEASUREMENTS.name}
                    GROUP BY station_abbr
                )
                INSERT INTO {ds.sa_table_x_station_var_availability.name}
                    (station_abbr, variable, value_count, min_date, max_date)
                {select_union_all}
            """
        )
        conn.execute(insert_sql)


def recreate_station_data_summary(engine: sa.Engine) -> None:
    """Creates a materialized view of summary data per station_abbr.

    The summary stats in this table are calculated across the whole dataset.
    """

    logger.info("Recreating station data summary")

    with engine.begin() as conn:
        # Drop old table if exists
        ds.sa_table_x_station_data_summary.drop(conn, checkfirst=True)

        # Create anew
        ds.sa_table_x_station_data_summary.create(conn)

        # Populate
        conn.execute(
            sa.text(
                f"""
            INSERT INTO {ds.sa_table_x_station_data_summary.name}
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

            FROM {ds.sa_table_smn_meta_stations.name} AS m
            LEFT JOIN (
                SELECT
                    station_abbr,
                    SUM(CASE WHEN tre200d0 IS NOT NULL THEN 1 END) AS tre200d0_count,
                    MIN(CASE WHEN tre200d0 IS NOT NULL THEN reference_timestamp END) AS tre200d0_min_date,
                    MAX(CASE WHEN tre200d0 IS NOT NULL THEN reference_timestamp END) AS tre200d0_max_date,
                    SUM(CASE WHEN rre150d0 IS NOT NULL THEN 1 END) AS rre150d0_count,
                    MIN(CASE WHEN rre150d0 IS NOT NULL THEN reference_timestamp END) AS rre150d0_min_date,
                    MAX(CASE WHEN rre150d0 IS NOT NULL THEN reference_timestamp END) AS rre150d0_max_date
                FROM {ds.TABLE_DAILY_MEASUREMENTS.name}
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
        FROM {ds.sa_table_x_station_data_summary.name}
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
        FROM {ds.sa_table_x_station_data_summary.name}
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
        FROM {ds.sa_table_x_nearby_stations.name}
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


_COLUMN_NAME_RE = re.compile(r"^[a-z][a-zA-Z0-9_]*$")


def _validate_column_names(columns: list[str]) -> None:
    """Checks that all elements of columns are valid SQL column names.

    Raises:
        ValueError if an invalid name if found.
    """
    if not all(_COLUMN_NAME_RE.search(c) for c in columns):
        raise ValueError(f"Invalid columns: {','.join(columns)}")


def read_daily_manual_measurements(
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
        columns = [
            dc.PRECIP_DAILY_MM,
            dc.SNOW_DEPTH_MAN_DAILY_CM,
            dc.FRESH_SNOW_MAN_DAILY_CM,
        ]

    _validate_column_names(columns)

    select_columns = ["station_abbr", "reference_timestamp"] + columns
    sql = f"""
        SELECT {', '.join(select_columns)}
        FROM {ds.TABLE_DAILY_MAN_MEASUREMENTS.name}
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
        columns = [
            dc.TEMP_DAILY_MEAN,
            dc.TEMP_DAILY_MIN,
            dc.TEMP_DAILY_MAX,
            dc.PRECIP_DAILY_MM,
        ]

    _validate_column_names(columns)

    select_columns = ["station_abbr", "reference_timestamp"] + columns
    sql = f"""
        SELECT {', '.join(select_columns)}
        FROM {ds.TABLE_DAILY_MEASUREMENTS.name}
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


def read_hourly_measurements(
    conn: sa.Connection,
    station_abbr: str,
    from_date: datetime.datetime,
    to_date: datetime.datetime,
    columns: list[str] | None = None,
    limit: int = -1,
) -> pd.DataFrame:
    if columns is None:
        columns = [
            dc.TEMP_HOURLY_MIN,
            dc.TEMP_HOURLY_MEAN,
            dc.TEMP_HOURLY_MAX,
            dc.PRECIP_HOURLY_MM,
        ]

    _validate_column_names(columns)

    select_columns = ["station_abbr", "reference_timestamp"] + columns
    sql = f"""
        SELECT {', '.join(select_columns)}
        FROM {ds.TABLE_HOURLY_MEASUREMENTS.name}
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
            dc.TEMP_MONTHLY_MIN,
            dc.TEMP_MONTHLY_MEAN,
            dc.TEMP_MONTHLY_MAX,
            dc.PRECIP_MONTHLY_MM,
        ]

    _validate_column_names(columns)

    select_columns = ["station_abbr", "reference_timestamp"] + columns
    sql = f"""
        SELECT {', '.join(select_columns)}
        FROM {ds.TABLE_MONTHLY_MEASUREMENTS.name}
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
        table=ds.var_summary_stats_all.sa_table,
        agg_name=agg_name,
        station_abbr=station_abbr,
        variables=variables,
    )
    if df.empty:
        return df
    # Drop the constant '*' time_slice dimension.
    return df.xs(dc.TS_ALL, level="time_slice")


def read_var_summary_stats_month(
    conn: sa.Connection,
    agg_name: str,
    station_abbr: str | None = None,
    months: list[int] | None = None,
    variables: Collection[str] | None = None,
) -> pd.DataFrame:
    return read_summary_stats(
        conn,
        table=ds.var_summary_stats_month.sa_table,
        agg_name=agg_name,
        time_slices=[dc.ts_month(m) for m in months] if months else None,
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
            ds.sa_table_x_station_var_summary_stats,
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


def read_measurement_infos(
    conn: sa.Connection, station_abbr: str
) -> list[models.MeasurementInfo]:

    # Parse possible date strings into actual dates (None stays None)
    def d(v: str | None) -> datetime.date | None:
        return datetime.date.fromisoformat(v) if v else None

    sql = f"""
            SELECT
                a.variable,
                a.value_count,
                a.min_date,
                a.max_date,
                p.parameter_description_de,
                p.parameter_description_fr,
                p.parameter_description_it,
                p.parameter_description_en,
                p.parameter_group_de,
                p.parameter_group_fr,
                p.parameter_group_it,
                p.parameter_group_en
            FROM {ds.sa_table_x_station_var_availability.name} AS a
            LEFT JOIN {ds.sa_table_smn_meta_parameters.name} AS p
                ON a.variable = p.parameter_shortname
            WHERE a.station_abbr = :station_abbr
            ORDER BY a.variable
        """

    result = conn.execute(sa.text(sql), {"station_abbr": station_abbr}).mappings().all()

    return [
        models.MeasurementInfo(
            variable=row["variable"],
            description=LocalizedString.from_nullable(
                de=row["parameter_description_de"],
                fr=row["parameter_description_fr"],
                it=row["parameter_description_it"],
                en=row["parameter_description_en"],
            ),
            group=LocalizedString.from_nullable(
                de=row["parameter_group_de"],
                fr=row["parameter_group_fr"],
                it=row["parameter_group_it"],
                en=row["parameter_group_en"],
            ),
            granularity="daily",
            value_count=row["value_count"],
            min_date=d(row["min_date"]),
            max_date=d(row["max_date"]),
        )
        for row in result
    ]


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
