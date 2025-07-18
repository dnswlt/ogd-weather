from concurrent.futures import ThreadPoolExecutor
import datetime
import glob
import os
import sqlite3
import pandas as pd
import requests
import sys
from urllib.parse import urlparse


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


def download_csv(url, output_dir, skip_existing=True):
    try:
        filename = os.path.basename(urlparse(url).path)
        output_path = os.path.join(output_dir, filename)

        if skip_existing and os.path.exists(output_path):
            print(f"Skipping existing file {output_path}")
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

    # Find "items' (data) and "assets" (metadata).
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
            if asset["href"].endswith("d_historical.csv"):
                csv_urls.append(asset["href"])

    print(f"Found {len(csv_urls)} CSV URLs. Example: {csv_urls[0]}")

    os.makedirs(weather_dir, exist_ok=True)
    for a in assets:
        download_csv(a["href"], output_dir=weather_dir)

    # Download CSV data files concurrently.
    def _download(url):
        download_csv(url, weather_dir)

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(_download, csv_urls)


def infer_sqlite_type(dtype: pd.api.types.CategoricalDtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "REAL"
    else:
        return "TEXT"


def normalize_timestamp(
    series: pd.Series, local_tz: str = "Europe/Zurich"
) -> pd.Series:
    """Normalizes a datetime Series for DB storage.

    - If all times are midnight, returns only "YYYY-MM-DD".
    - Otherwise assumes local_tz, converts to UTC, returns ISO with second granularity.
    """
    if series.empty:
        return series

    time_of_day = series.dt.time

    if time_of_day.nunique() == 1 and time_of_day.iloc[0] == datetime.time(0, 0):
        return series.dt.strftime("%Y-%m-%d")
    else:
        # Sub-daily measurements: use UTC
        return (
            series.dt.tz_localize(local_tz)
            .dt.tz_convert("UTC")
            .dt.strftime("%Y-%m-%d %H:%M:%S")
        )


def prepare_sql_ogd_smn_d_historical(dir: str, conn: sqlite3.Connection) -> None:
    table_name = "ogd_smn_d_historical"
    csv_pattern = "ogd-smn_*_d_historical.csv"

    # Create table if it doesn't exist (schema inferred from first CSV)

    files = glob.glob(os.path.join(dir, csv_pattern))
    if not files:
        print("No CSV files found.")
        return

    first_file = files[0]

    # Read one file to infer schema
    sample_df = pd.read_csv(
        first_file,
        sep=";",
        encoding="cp1252",
        parse_dates=["reference_timestamp"],
        date_format={"reference_timestamp": "%d.%m.%Y %H:%M"},
        nrows=10,
    )
    # Create table with proper PK
    cols_def = ",\n".join(
        f"{col} {infer_sqlite_type(dtype)}" for col, dtype in sample_df.dtypes.items()
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {cols_def},
            PRIMARY KEY (station_abbr, reference_timestamp)
        )
    """
    )
    conn.commit()

    # Process all files
    for fname in glob.glob(os.path.join(dir, csv_pattern)):
        print(f"Importing {fname} into sqlite3 database...")
        df = pd.read_csv(
            fname,
            sep=";",
            encoding="cp1252",
            parse_dates=["reference_timestamp"],
            date_format={"reference_timestamp": "%d.%m.%Y %H:%M"},
        )

        # Normalize timestamps: use YYYY-MM-DD if time is always 00:00
        df["reference_timestamp"] = normalize_timestamp(df["reference_timestamp"])

        # Insert row by row, ignoring duplicates
        placeholders = ",".join(["?"] * len(df.columns))
        insert_sql = f"INSERT OR IGNORE INTO {table_name} VALUES ({placeholders})"

        conn.executemany(insert_sql, df.itertuples(index=False, name=None))
        conn.commit()


def prepare_sql_ogd_smn_meta_stations(dir: str, conn: sqlite3.Connection) -> None:
    table_name = "ogd_smn_meta_stations"
    csv_file = os.path.join(dir, "ogd-smn_meta_stations.csv")

    df = pd.read_csv(
        csv_file,
        sep=";",
        encoding="cp1252",
        parse_dates=["station_data_since"],
        date_format={"station_data_since": "%d.%m.%Y"},
    )
    # Create table with proper PK
    cols_def = ",\n".join(
        f"{col} {infer_sqlite_type(dtype)}" for col, dtype in df.dtypes.items()
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {cols_def},
            PRIMARY KEY (station_abbr)
        )
    """
    )
    conn.commit()

    print(f"Importing {csv_file} into sqlite3 database...")

    # Normalize timestamps: use YYYY-MM-DD if time is always 00:00
    df["station_data_since"] = normalize_timestamp(df["station_data_since"])

    # Insert row by row, ignoring duplicates
    placeholders = ",".join(["?"] * len(df.columns))
    insert_sql = f"INSERT OR IGNORE INTO {table_name} VALUES ({placeholders})"

    conn.executemany(insert_sql, df.itertuples(index=False, name=None))
    conn.commit()


def prepare_sql_ogd_smn_station_data_summary(conn: sqlite3.Connection) -> None:
    # Drop old table
    conn.execute("DROP TABLE IF EXISTS ogd_smn_station_data_summary;")

    # Create it with proper schema & PK
    conn.execute(
        """
        CREATE TABLE ogd_smn_station_data_summary (
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

    # Single INSERT query → use execute()
    conn.execute(
        """
        INSERT INTO ogd_smn_station_data_summary
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

        FROM ogd_smn_meta_stations AS m

        LEFT JOIN (
            SELECT
                station_abbr,
                SUM(IIF(tre200d0 IS NOT NULL, 1, 0)) AS tre200d0_count,
                MIN(CASE WHEN tre200d0 IS NOT NULL THEN reference_timestamp END) AS tre200d0_min_date,
                MAX(CASE WHEN tre200d0 IS NOT NULL THEN reference_timestamp END) AS tre200d0_max_date,
                SUM(IIF(rre150d0 IS NOT NULL, 1, 0)) AS rre150d0_count,
                MIN(CASE WHEN rre150d0 IS NOT NULL THEN reference_timestamp END) AS rre150d0_min_date,
                MAX(CASE WHEN rre150d0 IS NOT NULL THEN reference_timestamp END) AS rre150d0_max_date
            FROM ogd_smn_d_historical
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

    Dates are stored as TEXT in ISO format "YYYY-MM-DD HH:MM:SS.SSS" and use UTC time.
    (This is different from the input CSV data, which uses local Swiss time!)

    Time is included only for sub-daily measurements, otherwise "YYYY-MM-DD" is used.
    In this case the stored date represents the local ("civil") date.

    The following data tables are supported:

    SwissMetNet (smn) historical daily measurements.

    * sqlite3 table name: ogd_smn_d_historical
    * File pattern: ogd-smn_*_d_historical.csv
    * Time resolution: daily
    * Primary key: (station_abbr, reference_timestamp)

    And the following metadata tables are supported:

    smn_meta_stations

    """
    db_path = os.path.join(dir, "swissmetnet.sqlite")
    conn = sqlite3.connect(db_path)

    # CSV data
    prepare_sql_ogd_smn_d_historical(dir, conn)
    prepare_sql_ogd_smn_meta_stations(dir, conn)
    # Materialized views
    prepare_sql_ogd_smn_station_data_summary(conn)

    conn.close()
    print(f"SQLite database updated at {db_path}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <output_dir>")
        sys.exit(1)
    weather_dir = sys.argv[1]
    fetch_all_data(weather_dir=weather_dir)
    prepare_sql_db(weather_dir)


if __name__ == "__main__":
    main()
