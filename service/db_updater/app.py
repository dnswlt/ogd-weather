import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import logging
import os
import re
import sqlalchemy as sa
import time
import pandas as pd
from pydantic import BaseModel
import requests
from urllib.parse import urlparse
from service.charts import db
from service.charts import logging_config as _  # configure logging


OGD_SNM_URL = (
    "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn"
)

UPDATE_STATUS_TABLE_NAME = "update_status"

logger = logging.getLogger("db_updater")


class CsvResource(BaseModel):
    href: str
    frequency: str = ""  # "historical", "recent", "now"
    interval: str = ""  # "d", "h", "" for metadata
    is_meta: bool = False
    updated: datetime.datetime | None = None
    status: db.UpdateStatus | None = None


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


def fetch_data_csv_resources(filter_regex: str | None = None) -> list[CsvResource]:
    """Fetch URLs of CSV data resources ("items")."""

    filter_re = re.compile(filter_regex) if filter_regex else None

    def _filter(href: str) -> bool:
        if filter_re is None:
            return True
        return bool(filter_re.search(href))

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
        # TODO: Re-enable _now_ data at some point.
        # It had a few missing points recently,
        # we don't reconcile with recent data yet, and we're currently
        # focusing on historical data anyway.
        mo = re.search(
            r".*_(d|h)_(historical|historical_2020-2029|recent|now__DISABLED__).csv$",
            href,
        )
        if not mo or not _filter(href):
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


def import_into_db(engine: sa.Engine, weather_dir: str, csvs: list[CsvResource]):
    """Imports all SwissMetNet (smn) CSV files into the sqlite3 database.

    Assumes that all files have already been downloaded to `weather_dir`.
    """

    def _fname(url):
        return os.path.basename(urlparse(url).path)

    daily = [c.status for c in csvs if c.interval == "d"]
    hourly = [c.status for c in csvs if c.interval == "h"]
    meta = [c.status for c in csvs if c.is_meta]

    if daily:
        db.insert_csv_data(
            weather_dir,
            engine,
            table_spec=db.TABLE_DAILY_MEASUREMENTS,
            csv_filenames=[_fname(s.href) for s in daily],
        )
        db.save_update_status(engine, daily)

    if hourly:
        db.insert_csv_data(
            weather_dir,
            engine,
            table_spec=db.TABLE_HOURLY_MEASUREMENTS,
            csv_filenames=[_fname(s.href) for s in hourly],
        )
        db.save_update_status(engine, hourly)
    if meta:
        table_map = {
            "ogd-smn_meta_parameters.csv": db.sa_table_meta_parameters,
            "ogd-smn_meta_stations.csv": db.sa_table_meta_stations,
        }
        for s in meta:
            csv_file = _fname(s.href)
            table = table_map.get(csv_file)
            if table is None:
                logger.warning("Ignoring unknown metadata CSV file %s", csv_file)
                continue
            db.insert_csv_metadata(
                engine,
                table=table,
                csv_file=os.path.join(weather_dir, csv_file),
            )
        db.save_update_status(engine, meta)


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

    # Update status fields. Create a new UpdateStatus ONLY if none
    # existed before. The existing ones must retain their DB ID.
    for c in update_csvs:
        if c.status and c.status.id is not None:
            # Reuse existing UpdateStatus → just update fields
            c.status.table_updated_time = now
            c.status.resource_updated_time = c.updated
            c.status.etag = etags.get(c.href)
        else:
            # No existing status or it's a brand-new one → create fresh
            c.status = db.UpdateStatus(
                id=None,  # Leave empty to process as an INSERT.
                href=c.href,
                table_updated_time=now,
                resource_updated_time=c.updated,
                etag=etags.get(c.href),
            )

    return update_csvs


def main():
    parser = argparse.ArgumentParser(
        description="Update weather DB.", allow_abbrev=False
    )
    parser.add_argument(
        "--base_dir",
        dest="base_dir",
        metavar="PATH",
        help="Directory for weather data (defaults to $OGD_BASE_DIR).",
    )
    parser.add_argument(
        "--csv_filter",
        dest="csv_filter",
        metavar="REGEX",
        help="Optional regular expression to update only a subset of data.",
    )
    parser.add_argument(
        "--recreate-views",
        action="store_true",
        help="Only recreate database views, do not update any data.",
    )

    args = parser.parse_args()

    if args.base_dir:
        base_dir = args.base_dir
    elif "OGD_BASE_DIR" in os.environ:
        base_dir = os.environ["OGD_BASE_DIR"]
    else:
        parser.error("weather_dir is required if $OGD_BASE_DIR is not set.")

    db_path = os.path.join(base_dir, db.DATABASE_FILENAME)
    logger.info("Connecting to sqlite DB at %s", db_path)

    started_time = time.time()

    sa_engine = sa.create_engine(f"sqlite:///{db_path}", echo=False)
    # Create tables if needed.
    db.metadata.create_all(sa_engine)

    if args.recreate_views:
        # Only recreate views, then exit.
        db.recreate_views(sa_engine)
        logger.info(
            "Recreated materialized views in %.1fs.", time.time() - started_time
        )
        return

    data_csvs = fetch_data_csv_resources(args.csv_filter)
    meta_csvs = fetch_metadata_csv_resources()
    csvs = meta_csvs + data_csvs

    statuses = db.read_update_status(sa_engine)
    logger.debug("Fetched %d update statuses from DB", len(statuses))

    # Assign statuses to corresponding CsvResource.
    status_map = {s.href: s for s in statuses}
    for c in csvs:
        c.status = status_map.get(c.href)

    updated_csvs = fetch_latest_data(base_dir, csvs)
    if updated_csvs:
        import_into_db(sa_engine, base_dir, updated_csvs)

    # Recreate materialized views
    logger.info("Recreating materialized views...")
    db.recreate_views(sa_engine)

    logger.info("Done. DB updated in %.1fs.", time.time() - started_time)


if __name__ == "__main__":
    main()
