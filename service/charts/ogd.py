from concurrent.futures import ThreadPoolExecutor
import glob
import os
import re
import sqlite3
import time
import numpy as np
import pandas as pd
import requests
import sys
from urllib.parse import urlparse
from . import db
from . import logging_config as _  # configure logging

_DATE_FORMATS = {
    "station_data_since": "%d.%m.%Y",
    "reference_timestamp": "%d.%m.%Y %H:%M",
}


def fetch_paginated(url, timeout=10):
    """Yields paginated resources from a web API that uses HAL-style links."""
    next_url = url

    while next_url:
        try:
            resp = requests.get(next_url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            raise RuntimeError(f"Failed to fetch or parse {next_url}: {e}") from e

        yield data

        links = data.get("links", [])
        if not isinstance(links, list):
            break
        next_url = next(
            (link.get("href") for link in links if link.get("rel") == "next"), None
        )


def download_csv(url, output_dir, max_age_seconds: int | None = None):
    """Downloads a CSV file from the given URL url and writes it to output_dir.

    Args:
        * url - the URL to download from
        * output_dir - the directory to which the file gets written, using
            the same filename as the url.
        * max_age_seconds - maximum age of an existing file (based on its mtime)
            for it to get skipped if skip_existing=True.
    """
    try:
        filename = os.path.basename(urlparse(url).path)
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            if max_age_seconds is None:
                # Always overwrite
                pass
            elif max_age_seconds < 0:
                # Infinite max age: never overwrite
                print(f"Skipping existing file {output_path}")
                return
            elif time.time() - os.path.getmtime(output_path) < max_age_seconds:
                print(f"Skipping existing fresh file {output_path}")
                return

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded: {filename}")
    except Exception as e:
        print(f"Failed: {url} — {e}")


def fetch_all_data(weather_dir: str):
    collections = []
    for r in fetch_paginated("https://data.geo.admin.ch/api/stac/v1/collections"):
        collections.extend(r["collections"])

    print(f"Read {len(collections)} collections.")
    cs = {c["id"]: c for c in collections}

    # Find "items" (data) and "assets" (metadata).
    assets_url = None
    items_url = None
    for l in cs["ch.meteoschweiz.ogd-smn"]["links"]:
        if l["rel"] == "assets":
            assets_url = l["href"]
        elif l["rel"] == "items":
            items_url = l["href"]

    assets = []
    for r in fetch_paginated(assets_url):
        assets.extend(r["assets"])
    print(f"Read {len(assets)} assets.")

    features = []
    for r in fetch_paginated(items_url):
        features.extend(r["features"])
    print(f"Read {len(features)} features.")

    csv_urls = []
    for feature in features:
        for asset in feature["assets"].values():
            url = asset["href"]
            if re.search(r".*_(d_historical|h_historical_2020-2029).csv$", url):
                csv_urls.append((url, -1))  # -1: Skip existing CSV files.
            elif re.search(r".*_(d|h)_recent.csv$", url):
                # Overwrite existing for recent data every 12 hours.
                csv_urls.append((url, 12 * 60 * 60))

    print(f"Found {len(csv_urls)} CSV URLs. Example: {csv_urls[0][0]}")

    os.makedirs(weather_dir, exist_ok=True)
    for a in assets:
        # Download metadata at most once a day
        download_csv(a["href"], output_dir=weather_dir, max_age_seconds=24 * 60 * 60)

    # Download CSV data files concurrently.
    def _download(params):
        url, max_age_seconds = params
        # Download fresh data at most every 12h.
        download_csv(url, weather_dir, max_age_seconds=max_age_seconds)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(_download, csv_urls)


def infer_sqlite_type(dtype: pd.api.types.CategoricalDtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "REAL"
    else:
        return "TEXT"


def determine_column_types(csv_file: str):
    """Returns all columns present in both _DATE_FORMATS and the given CSV file."""
    df = pd.read_csv(
        csv_file,
        sep=";",
        encoding="cp1252",
        nrows=1,
    )

    def is_measurement(c):
        # Heuristically determine if the column is a measurement column.
        return len(c) == len("tre200h0") and re.search(r"^[a-z0-9]+$", c)

    return {
        "date_format": {
            c: fmt for (c, fmt) in _DATE_FORMATS.items() if c in df.columns
        },
        "dtype": {c: np.float64 for c in df.columns if is_measurement(c)},
    }


def prepare_sql_table(
    dir: str,
    conn: sqlite3.Connection,
    table_name: str,
    csv_pattern: str,
    primary_keys: list[str] | None = None,
) -> None:
    # Create table if it doesn't exist (schema inferred from first CSV)
    files = glob.glob(os.path.join(dir, csv_pattern))
    if not files:
        print(f"No CSV files matching {csv_pattern} found.")
        return

    # Pandas expects all columns passed to read_csv's parse_dates= to exist.
    # Identify which columns exist by reading the first file:
    typeinfo = determine_column_types(files[0])
    date_format = typeinfo["date_format"]

    schema_created = False
    for fname in files:
        print(f"Importing {fname} into sqlite3 database...")
        df = pd.read_csv(
            fname,
            sep=";",
            encoding="cp1252",
            date_format=date_format,
            parse_dates=list(date_format.keys()),
            dtype=typeinfo["dtype"],
        )
        # Normalize timestamps: use YYYY-MM-DD if time is always 00:00
        for col in date_format.keys():
            df[col] = db.normalize_timestamp(df[col])

        if not schema_created:
            # Create table with proper PK
            cols_def = ",\n".join(
                f"{col} {infer_sqlite_type(dtype)}" for col, dtype in df.dtypes.items()
            )
            primary_key = (
                f"PRIMARY KEY ({', '.join(primary_keys)})" if primary_keys else ""
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {cols_def},
                    {primary_key}
                )
            """
            )
            conn.commit()
            schema_created = True

        # Insert row by row, ignoring duplicates
        placeholders = ",".join(["?"] * len(df.columns))
        insert_sql = f"INSERT OR IGNORE INTO {table_name} VALUES ({placeholders})"

        conn.executemany(insert_sql, df.itertuples(index=False, name=None))
        conn.commit()


def import_sql_ogd_smn_hourly(dir: str, conn: sqlite3.Connection) -> None:
    db.prepare_sql_table_from_spec(
        dir,
        conn,
        table_spec=db.HOURLY_MEASUREMENTS_TABLE,
        csv_pattern="ogd-smn_*_h_historical_*.csv",
    )
    db.prepare_sql_table_from_spec(
        dir,
        conn,
        table_spec=db.HOURLY_MEASUREMENTS_TABLE,
        csv_pattern="ogd-smn_*_h_recent.csv",
    )


def import_sql_ogd_smn_daily(dir: str, conn: sqlite3.Connection) -> None:
    db.prepare_sql_table_from_spec(
        dir,
        conn,
        table_spec=db.DAILY_MEASUREMENTS_TABLE,
        csv_pattern="ogd-smn_*_d_historical.csv",
    )
    db.prepare_sql_table_from_spec(
        dir,
        conn,
        table_spec=db.DAILY_MEASUREMENTS_TABLE,
        csv_pattern="ogd-smn_*_d_recent.csv",
    )


def import_sql_ogd_smn_meta_stations(dir: str, conn: sqlite3.Connection) -> None:
    prepare_sql_table(
        dir,
        conn,
        table_name=db.META_STATIONS_TABLE_NAME,
        csv_pattern=os.path.join(dir, "ogd-smn_meta_stations.csv"),
        primary_keys=["station_abbr"],
    )


def import_sql_ogd_smn_meta_parameters(dir: str, conn: sqlite3.Connection) -> None:
    prepare_sql_table(
        dir,
        conn,
        table_name=db.META_PARAMETERS_TABLE_NAME,
        csv_pattern=os.path.join(dir, "ogd-smn_meta_parameters.csv"),
        primary_keys=["parameter_shortname"],
    )


def recreate_station_data_summary(conn: sqlite3.Connection) -> None:
    """Creates a materialized view of summary data per station_abbr.

    The data is based on a JOIN of the ogd_smn_meta_stations and ogd_smn_d_historical tables.
    """
    # Drop old table
    conn.execute(f"DROP TABLE IF EXISTS {db.STATION_DATA_SUMMARY_TABLE_NAME};")

    # Create it with proper schema & PK
    conn.execute(
        f"""
        CREATE TABLE {db.STATION_DATA_SUMMARY_TABLE_NAME} (
            station_abbr TEXT PRIMARY KEY,
            station_name TEXT,
            station_canton TEXT,
            station_wigos_id TEXT,
            station_type_en TEXT,
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
        INSERT INTO {db.STATION_DATA_SUMMARY_TABLE_NAME}
        SELECT
            m.station_abbr,
            m.station_name,
            m.station_canton,
            m.station_wigos_id,
            m.station_type_en,
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

        FROM {db.META_STATIONS_TABLE_NAME} AS m

        LEFT JOIN (
            SELECT
                station_abbr,
                SUM(IIF(tre200d0 IS NOT NULL, 1, 0)) AS tre200d0_count,
                MIN(CASE WHEN tre200d0 IS NOT NULL THEN reference_timestamp END) AS tre200d0_min_date,
                MAX(CASE WHEN tre200d0 IS NOT NULL THEN reference_timestamp END) AS tre200d0_max_date,
                SUM(IIF(rre150d0 IS NOT NULL, 1, 0)) AS rre150d0_count,
                MIN(CASE WHEN rre150d0 IS NOT NULL THEN reference_timestamp END) AS rre150d0_min_date,
                MAX(CASE WHEN rre150d0 IS NOT NULL THEN reference_timestamp END) AS rre150d0_max_date
            FROM {db.DAILY_MEASUREMENTS_TABLE.name}
            GROUP BY station_abbr
        ) AS h
        ON m.station_abbr = h.station_abbr;
    """
    )

    conn.commit()


def prepare_sql_db(dir: str) -> None:
    """Loads all SwissMetNet (smn) CSV files into a single sqlite3 database for fast and uniform access.

    Column names in the DB tables are identical to column names in the CSV (e.g.,
    station_abbr, reference_timestamp, tre200d0).

    Dates are stored as TEXT in ISO format "YYYY-MM-DD HH:MM:SSZ" and use UTC time.

    Time is included only for sub-daily measurements, otherwise "YYYY-MM-DD" is used.

    The following data tables are supported:

    * SwissMetNet (smn) daily measurements.

        * sqlite3 table name: ogd_smn_daily
        * File patterns: ogd-smn_*_d_historical.csv
        * Time resolution: daily
        * Primary key: (station_abbr, reference_timestamp)

    * SwissMetNet (smn) recent hourly measurements.

        * sqlite3 table name: ogd_smn_hourly
        * File patterns: ogd-smn_*_h_recent.csv
        * Time resolution: hourly
        * Primary key: (station_abbr, reference_timestamp)

    * And the following metadata tables are supported:

        * smn_meta_stations
        * smn_meta_parameters

    """
    db_path = os.path.join(dir, db.DATABASE_FILENAME)
    conn = sqlite3.connect(db_path)

    # CSV data
    import_sql_ogd_smn_daily(dir, conn)
    import_sql_ogd_smn_hourly(dir, conn)
    import_sql_ogd_smn_meta_stations(dir, conn)
    import_sql_ogd_smn_meta_parameters(dir, conn)
    # Materialized views
    recreate_station_data_summary(conn)

    conn.close()
    print(f"SQLite database updated at {db_path}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <output_dir>")
        sys.exit(1)
    weather_dir = sys.argv[1]
    fetch_all_data(weather_dir)
    prepare_sql_db(weather_dir)


if __name__ == "__main__":
    main()
