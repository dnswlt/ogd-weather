import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import logging
import os
import queue
import sqlalchemy as sa
import time
from pydantic import BaseModel
import requests
from urllib.parse import urlparse
from service.charts import db
from service.charts import logging_config as _  # configure logging

from .smn import match_csv_resource

OGD_SNM_URL = (
    "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn"
)

UPDATE_STATUS_TABLE_NAME = "update_status"

_NOOP_SENTINEL = object()

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


def fetch_data_csv_resources(filter_re: str | None = None) -> list[CsvResource]:
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
        match = match_csv_resource(href, filter_re)
        if not match:
            continue
        if match.years and match.years[0] < 1990 and match.interval == "h":
            # Only download hourly historical data from 1990 onwards.
            # (Keep data volume at bay, and many stations don't have older data anyway)
            continue

        updated = (
            datetime.datetime.fromisoformat(asset["updated"])
            if asset["updated"]
            else None
        )

        resources.append(
            CsvResource(
                href=href,
                updated=updated,
                frequency=match.frequency,
                interval=match.interval,
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


def import_into_db(
    engine: sa.Engine,
    weather_dir: str,
    resource: CsvResource,
    insert_mode: str,
):
    """Imports a SwissMetNet (smn) CSV file into the database.

    Assumes that the file has already been downloaded to `weather_dir`.
    """

    measurement_tables = {
        "h": db.TABLE_HOURLY_MEASUREMENTS,
        "d": db.TABLE_DAILY_MEASUREMENTS,
        "m": db.TABLE_MONTHLY_MEASUREMENTS,
    }
    metadata_tables = {
        "ogd-smn_meta_parameters.csv": db.sa_table_meta_parameters,
        "ogd-smn_meta_stations.csv": db.sa_table_meta_stations,
    }

    if resource.interval in measurement_tables:
        db.insert_csv_data(
            weather_dir,
            engine,
            table_spec=measurement_tables[resource.interval],
            update=resource.status,
            insert_mode=insert_mode,
        )
    elif resource.is_meta:
        csv_file = resource.status.filename()
        table = metadata_tables.get(csv_file)
        if table is None:
            logger.warning("Ignoring unknown metadata CSV file %s", csv_file)
            return
        db.insert_csv_metadata(
            weather_dir,
            engine,
            table=table,
            update=resource.status,
        )
    else:
        raise AssertionError(f"Unhandled case: {resource}")


def fetch_latest_data(
    weather_dir: str,
    csvs: list[CsvResource],
    executor: ThreadPoolExecutor,
    import_queue: queue.SimpleQueue,
    force_update: bool = False,
) -> list[CsvResource]:
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
        if force_update:
            logger.debug("Adding %s (forced)", c.href)
            update_csvs.append(c)
            continue

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
    def _download_and_queue(c: CsvResource) -> str | None:
        try:
            etag = download_csv(
                c.href,
                weather_dir,
                etag=c.status.etag if c.status and not force_update else None,
            )
        except Exception as e:
            logger.error("Failed to download %s: %s", c.href, str(e))
            import_queue.put(_NOOP_SENTINEL)  # Inform consumer that we're done.
            return

        if c.status and c.status.id is not None:
            # Reuse existing UpdateStatus → just update fields
            c.status.table_updated_time = now
            c.status.resource_updated_time = c.updated
            c.status.etag = etag
        else:
            # No existing status or it's a brand-new one → create fresh
            c.status = db.UpdateStatus(
                id=None,  # Leave empty to process as an INSERT.
                href=c.href,
                table_updated_time=now,
                resource_updated_time=c.updated,
                etag=etag,
            )
        import_queue.put(c)

    futures = []
    for u in update_csvs:
        fut = executor.submit(_download_and_queue, u)
        futures.append(fut)

    return futures


def main():
    parser = argparse.ArgumentParser(
        description="Update weather DB.", allow_abbrev=False
    )
    parser.add_argument(
        "--base-dir",
        dest="base_dir",
        metavar="PATH",
        help="Directory for weather data (defaults to $OGD_BASE_DIR).",
    )
    parser.add_argument(
        "--postgres-url",
        dest="postgres_url",
        metavar="URL",
        help=(
            "Connection URL for Postgres "
            "(e.g., postgresql+psycopg://user@host:port/dbname). "
            "Avoid hardcoding the password — use ~/.pgpass or PGPASSWORD env variable."
        ),
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        default=False,
        help="If set, ALL DATA IS DROPPED from the database before updating it.",
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        default=False,
        help="If set, all downloaded CSV data is upserted into the database (even if data already exists).",
    )
    parser.add_argument(
        "--csv-filter",
        dest="csv_filter",
        metavar="REGEX",
        help="Optional regular expression to update only a subset of data.",
    )
    parser.add_argument(
        "--recreate-views",
        action="store_true",
        default=False,
        help="Only recreate database views, do not update any data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="More verbose logging (e.g. SQLAlchemy statements)",
    )

    args = parser.parse_args()

    if args.base_dir:
        base_dir = args.base_dir
    elif "OGD_BASE_DIR" in os.environ:
        base_dir = os.environ["OGD_BASE_DIR"]
    else:
        parser.error("base_dir is required if $OGD_BASE_DIR is not set.")

    postgres_url = args.postgres_url or os.environ.get("OGD_POSTGRES_URL")

    started_time = time.time()

    if postgres_url:
        logger.info("Connecting to postgres DB at %s", postgres_url)
        engine = sa.create_engine(postgres_url, echo=args.verbose)
    else:
        db_path = os.path.join(base_dir, db.DATABASE_FILENAME)
        logger.info("Connecting to sqlite DB at %s", db_path)
        engine = sa.create_engine(f"sqlite:///{db_path}", echo=args.verbose)

    logger.info("Using DB engine '%s'", engine.name)

    if args.recreate_views:
        # Only recreate views, then exit.
        db.recreate_views(engine)
        logger.info(
            "Recreated materialized views in %.1fs.", time.time() - started_time
        )
        return

    force_update: bool = args.force_update
    force_recreate: bool = args.force_recreate

    # Drop old data if requested.
    if force_recreate:
        logger.warning("Dropping existing data from all tables.")
        db.metadata.drop_all(bind=engine)

    if force_update:
        logger.info("--force-update was set, forcing data update for all assets")

    # Determine insert mode.
    insert_mode = "insert_missing"
    if force_recreate:
        insert_mode = "append"
    elif force_update:
        insert_mode = "merge"

    # Create tables if needed.
    db.metadata.create_all(engine)

    # Fetch CSV URLs.
    data_csvs = fetch_data_csv_resources(args.csv_filter)
    meta_csvs = fetch_metadata_csv_resources()
    csvs = meta_csvs + data_csvs

    statuses = db.read_update_status(engine)
    logger.debug("Fetched %d update statuses from DB", len(statuses))

    # Assign statuses to corresponding CsvResource.
    status_map = {s.href: s for s in statuses}
    for c in csvs:
        c.status = status_map.get(c.href)

    # Now download CSVs concurrently and feed them to the importer thread
    # via a queue:
    import_queue = queue.SimpleQueue()
    with ThreadPoolExecutor(max_workers=8) as executor:

        futures = fetch_latest_data(
            base_dir,
            csvs,
            executor=executor,
            import_queue=import_queue,
            force_update=force_update,
        )
        # Receive UpdateStatus items to import into the DB:
        for _ in range(len(futures)):
            work_item = import_queue.get()
            if work_item is _NOOP_SENTINEL:
                continue  # Skip failed CSV imports
            resource: CsvResource = work_item
            import_into_db(engine, base_dir, resource, insert_mode=insert_mode)

        # Consume futures (they're all done)
        for fut in as_completed(futures):
            fut.result()  # Raise exception if any occurred (indicates programming error)

    # Recreate materialized views
    logger.info("Recreating materialized views...")
    db.recreate_views(engine)

    logger.info("Done. DB %s updated in %.1fs.", engine.url, time.time() - started_time)


if __name__ == "__main__":
    main()
