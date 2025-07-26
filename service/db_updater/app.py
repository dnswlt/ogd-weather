import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import logging
import os
import re
import sqlite3
import time
from typing import Iterable
import pandas as pd
from pydantic import BaseModel
import requests
import sys
from urllib.parse import urlparse
from service.charts import db
from service.charts import logging_config as _  # configure logging


OGD_SNM_URL = (
    "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn"
)

UPDATE_STATUS_TABLE_NAME = "update_status"

logger = logging.getLogger("db_updater")


class UpdateStatus(BaseModel):
    href: str
    table_updated_time: datetime.datetime
    resource_updated_time: datetime.datetime | None = None
    etag: str | None = None


class CsvResource(BaseModel):
    href: str
    frequency: str = ""  # "historical", "recent", "now"
    interval: str = ""  # "d", "h", "" for metadata
    is_meta: bool = False
    updated: datetime.datetime | None = None
    status: UpdateStatus | None = None


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


def download_csv(url, output_dir, etag: str | None = None) -> str | None:
    """Downloads a CSV file from the given URL url and writes it to output_dir.

    Args:
        * url - the URL to download from
        * output_dir - the directory to which the file gets written, using
            the same filename as the url.
        * etag - an optional Etag to pass in the http header. On a 304 response
            no file will be downloaded.

    Returns:
        The Etag header, if present.
    """
    filename = os.path.basename(urlparse(url).path)
    output_path = os.path.join(output_dir, filename)

    headers = {}
    if etag:
        headers["If-None-Match"] = etag
    response = requests.get(url, timeout=30, headers=headers)
    response.raise_for_status()
    if response.status_code == 304:
        logger.info("Skipping %s: 304 Not Modified", filename)
        return etag

    etag = response.headers.get("Etag")

    with open(output_path, "wb") as f:
        f.write(response.content)

    logger.info(f"Downloaded: {filename}")
    return etag


def fetch_data_csv_resources() -> list[CsvResource]:
    """Fetch URLs of CSV data resources ("items")."""

    # Fetch assets from /items
    assets = []
    for r in fetch_paginated(f"{OGD_SNM_URL}/items"):
        for feature in r.get("features", []):
            for asset in feature.get("assets", {}).values():
                assets.append(asset)

    # Transform asset features
    resources = []
    for asset in assets:
        href = asset["href"]
        mo = re.search(
            r".*_(d|h)_(historical|historical_2020-2029|recent|now).csv$", href
        )
        if not mo:
            continue

        updated = (
            datetime.datetime.fromisoformat(asset["updated"])
            if asset["updated"]
            else None
        )

        interval = mo.group(1)
        if interval not in ["d", "h"]:
            raise ValueError(f"Unknown interval {interval}")

        freq = mo.group(2).split("_")[0]
        if freq not in ["historical", "recent", "now"]:
            raise ValueError(f"Failed to identify freshness of {href}: {freq}")

        resources.append(
            CsvResource(
                href=href,
                updated=updated,
                frequency=freq,
                interval=interval,
            )
        )
    return resources


def fetch_metadata_csv_resources() -> list[CsvResource]:
    """Fetch URLs of metadata CSV resources ("assets")."""
    resources = []
    for r in fetch_paginated(f"{OGD_SNM_URL}/assets"):
        for asset in r.get("assets", []):
            updated = (
                datetime.datetime.fromisoformat(asset["updated"])
                if asset["updated"]
                else None
            )
            # Treat Metadata as historical, i.e. with low update frequency.
            resources.append(
                CsvResource(
                    href=asset["href"],
                    updated=updated,
                    interval="",
                    frequency="historical",
                    is_meta=True,
                )
            )
    return resources


def infer_sqlite_type(dtype: pd.api.types.CategoricalDtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "REAL"
    else:
        return "TEXT"


def prepare_metadata_table(
    conn: sqlite3.Connection,
    csv_file: str,
    table_name: str,
) -> None:
    """Creates (if not exists) metadata table `table_name`.

    All columns and their types are inferred from the CSV file.
    """
    logger.info(f"Importing {csv_file} into sqlite3 database...")
    df = pd.read_csv(
        csv_file,
        sep=";",
        encoding="cp1252",
    )
    # Create table with proper PK
    cols_def = ",\n".join(
        f"{col} {infer_sqlite_type(dtype)}" for col, dtype in df.dtypes.items()
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {cols_def}
        )
    """
    )
    conn.commit()

    # Insert row by row, ignoring duplicates
    placeholders = ",".join(["?"] * len(df.columns))
    insert_sql = f"INSERT OR IGNORE INTO {table_name} VALUES ({placeholders})"

    conn.executemany(insert_sql, df.itertuples(index=False, name=None))
    conn.commit()


def create_update_status(conn: sqlite3.Connection) -> None:
    """Creates the "update_status" table if it does not exist."""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {UPDATE_STATUS_TABLE_NAME} (
            href TEXT PRIMARY KEY,
            table_updated_time TEXT NOT NULL,
            resource_updated_time TEXT,
            etag TEXT
        );
    """
    )
    conn.commit()


def update_update_status(
    conn: sqlite3.Connection, statuses: Iterable[UpdateStatus]
) -> None:
    """Inserts or updates the statuses in the update status table."""
    tuples = []
    for s in statuses:
        table_updated_time = (
            s.table_updated_time.isoformat() if s.table_updated_time else None
        )
        resource_updated_time = (
            s.resource_updated_time.isoformat() if s.resource_updated_time else None
        )
        tuples.append((s.href, table_updated_time, resource_updated_time, s.etag))

    insert_sql = f"""
        INSERT OR REPLACE INTO {UPDATE_STATUS_TABLE_NAME}
            (href, table_updated_time, resource_updated_time, etag)
        VALUES (?, ?, ?, ?)
    """
    conn.executemany(insert_sql, tuples)
    conn.commit()


def read_update_status(conn: sqlite3.Connection) -> list[UpdateStatus]:
    """Reads all rows from the update status table."""
    c = conn.execute(
        f"""
        SELECT href, resource_updated_time, table_updated_time, etag
        FROM {UPDATE_STATUS_TABLE_NAME}
    """
    )
    statuses = []

    def d(s: str | None) -> datetime.date | None:
        return datetime.datetime.fromisoformat(s) if s else None

    for row in c.fetchall():
        statuses.append(
            UpdateStatus(
                href=row["href"],
                resource_updated_time=d(row["resource_updated_time"]),
                table_updated_time=d(row["table_updated_time"]),
                etag=row["etag"],
            )
        )
    logger.debug("Fetched %d update statuses from DB", len(statuses))
    return statuses


def import_into_db(conn: sqlite3.Connection, weather_dir: str, csvs: list[CsvResource]):
    """Imports all SwissMetNet (smn) CSV files into the sqlite3 database.

    Assumes that all files have already been downloaded to `weather_dir`.
    """

    def fname(url):
        return os.path.basename(urlparse(url).path)

    daily_files = [fname(c.href) for c in csvs if c.interval == "d"]
    hourly_files = [fname(c.href) for c in csvs if c.interval == "h"]
    meta_files = [fname(c.href) for c in csvs if c.is_meta]

    if daily_files:
        db.prepare_sql_table_from_spec(
            weather_dir,
            conn,
            table_spec=db.DAILY_MEASUREMENTS_TABLE,
            csv_filenames=daily_files,
        )
    if hourly_files:
        db.prepare_sql_table_from_spec(
            weather_dir,
            conn,
            table_spec=db.HOURLY_MEASUREMENTS_TABLE,
            csv_filenames=hourly_files,
        )
    if meta_files:
        table_map = {
            "ogd-smn_meta_parameters.csv": db.META_PARAMETERS_TABLE_NAME,
            "ogd-smn_meta_stations.csv": db.META_STATIONS_TABLE_NAME,
        }
        for meta_file in meta_files:
            table_name = table_map.get(meta_file)
            if not table_name:
                logger.warning("Ignoring unknown metadata CSV file %s", meta_file)
                continue
            prepare_metadata_table(
                conn,
                csv_file=os.path.join(weather_dir, meta_file),
                table_name=table_name,
            )


def fetch_latest_data(weather_dir: str, csvs: list[CsvResource]) -> list[CsvResource]:
    now = datetime.datetime.now()

    def should_refresh(c: CsvResource) -> bool:
        t = c.status.table_updated_time
        f = c.frequency
        if f == "historical":
            # Try once per week to get new historical data.
            return (now - t).days >= 7
        elif f == "recent":
            # Try every 6 hours
            return (now - t).total_seconds() >= 6 * 60 * 60
        elif f == "now":
            # Try every 10 minutes
            return (now - t).total_seconds() >= 10 * 60
        return False

    # Determine which CSV resources to update.
    update_csvs: list[CsvResource] = []
    for c in csvs:
        if c.status is None:
            logger.debug("Adding %s (new entry)", c.href)
            update_csvs.append(c)
            continue

        if c.status.table_updated_time is None:
            logger.debug("Adding %s (no table_updated_time)", c.href)
            update_csvs.append(c)
        elif (
            c.updated
            and c.status.resource_updated_time
            and c.updated > c.status.resource_updated_time
        ):
            logger.debug("Adding %s (resource updated)", c.href)
            update_csvs.append(c)
        elif should_refresh(c):
            logger.debug("Adding %s (should_refresh)", c.href)
            update_csvs.append(c)

    if not update_csvs:
        logger.info("No new CSV files to download.")
        return []

    logger.info("Downloading %d CSV files...", len(update_csvs))

    # Download CSV data files concurrently.
    def _download(c: CsvResource) -> str | None:
        try:
            etag = download_csv(
                c.href, weather_dir, etag=c.status.etag if c.status else None
            )
            return (c, etag)
        except Exception as e:
            logger.error("Failed to download %s: %s", c.href, str(e))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(_download, update_csvs))
        etags = {c.href: etag for (c, etag) in results}

    # Update status fields.
    for c in update_csvs:
        c.status = UpdateStatus(
            href=c.href,
            table_updated_time=now,
            resource_updated_time=c.updated,
            etag=etags.get(c.href),
        )

    return update_csvs


def recreate_views(conn: sqlite3.Connection) -> None:
    db.recreate_station_data_summary(conn)
    db.recreate_ref_period_1991_2020_stats(conn)


def main():
    parser = argparse.ArgumentParser(
        description="Update weather DB.", allow_abbrev=False
    )
    parser.add_argument(
        "weather_dir",
        nargs="?",  # optional, can come from env
        help="Directory for weather data (defaults to $OGD_BASE_DIR).",
    )
    parser.add_argument(
        "--recreate-views",
        action="store_true",
        help="Only recreate database views, do not update any data.",
    )

    args = parser.parse_args()

    if args.weather_dir:
        weather_dir = args.weather_dir
    elif "OGD_BASE_DIR" in os.environ:
        weather_dir = os.environ["OGD_BASE_DIR"]
    else:
        parser.error("weather_dir is required if $OGD_BASE_DIR is not set.")

    db_path = os.path.join(weather_dir, db.DATABASE_FILENAME)
    logger.info("Connecting to sqlite DB at %s", db_path)

    started_time = time.time()

    if args.recreate_views:
        with sqlite3.connect(db_path) as conn:
            # Only recreate views, then exit.
            recreate_views(conn)
            logger.info(
                "Recreated materialized views in %.1fs.", time.time() - started_time
            )
            return

    with sqlite3.connect(db_path) as conn:

        conn.row_factory = sqlite3.Row
        create_update_status(conn)

        data_csvs = fetch_data_csv_resources()
        meta_csvs = fetch_metadata_csv_resources()
        csvs = meta_csvs + data_csvs

        statuses = read_update_status(conn)
        # Assign statuses to corresponding CsvResource.
        status_map = {s.href: s for s in statuses}
        for c in csvs:
            c.status = status_map.get(c.href)

        updated_csvs = fetch_latest_data(weather_dir, csvs)

        import_into_db(conn, weather_dir, updated_csvs)

        # Update status after successful import
        update_update_status(conn, [c.status for c in updated_csvs])

        # Recreate materialized views
        logger.info("Recreating materialized views...")
        recreate_views(conn)

    logger.info("Done. DB updated in %.1fs.", time.time() - started_time)


if __name__ == "__main__":
    main()
