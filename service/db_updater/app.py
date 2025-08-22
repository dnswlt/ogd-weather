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

from service.charts.base import logging_config as _  # configure logging

from service.charts.db.dbconn import PgConnectionInfo
from service.charts.db import db
from service.charts.db import schema as ds
from service.charts.db import constants as dc

from .bootstrap import bootstrap_postgres
from .smn import match_csv_resource

OGD_SNM_URL = (
    "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn"
)

OGD_NIME_URL = (
    "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-nime"
)

UPDATE_STATUS_TABLE_NAME = "update_status"

_NOOP_SENTINEL = object()

logger = logging.getLogger("db_updater")


class CsvResource(BaseModel):
    parent_id: str  # E.g. "ch.meteoschweiz.ogd-smn"
    href: str
    frequency: str = ""  # "historical", "recent", "now"
    interval: str = ""  # "d", "h", "" for metadata
    is_meta: bool = False
    updated_time: datetime.datetime | None = None
    status: db.UpdateStatus | None = None


def fetch_single(url, timeout=10):
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as e:
        raise RuntimeError(f"Failed to fetch or parse {url}: {e}") from e


def fetch_paginated(url, timeout=10):
    """Yields paginated resources from a web API that uses HAL-style links."""
    next_url = url

    while next_url:
        data = fetch_single(next_url, timeout)

        yield data

        links = data.get("links", [])
        if not isinstance(links, list):
            break
        next_url = next(
            (link.get("href") for link in links if link.get("rel") == "next"), None
        )


def download_csv(url, output_dir, etag: str | None = None) -> tuple[str | None, bool]:
    """Downloads a CSV file from the given URL url and writes it to output_dir.

    Args:
        * url - the URL to download from
        * output_dir - the directory to which the file gets written, using
            the same filename as the url.
        * etag - an optional Etag to pass in the http header. On a 304 response
            no file will be downloaded.

    Returns:
        A tuple of (ETag, is_modified).
            is_modified is False if the server returned 304 Not Modified.
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
        return (etag, False)

    etag = response.headers.get("Etag")

    with open(output_path, "wb") as f:
        f.write(response.content)

    logger.info(f"Downloaded: {filename}")
    return (etag, True)


def fetch_data_csv_resources(
    base_url: str, filter_re: str | None = None
) -> list[CsvResource]:
    """Fetch URLs of CSV data resources ("items")."""

    # Fetch parent resource to get its ID
    # (we could just use the last part of the URL)
    res = fetch_single(base_url)
    parent_id = res["id"]

    # Fetch assets from /items
    assets = []
    for r in fetch_paginated(f"{base_url}/items"):
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
                parent_id=parent_id,
                href=href,
                updated_time=updated,
                frequency=match.frequency,
                interval=match.interval,
            )
        )
    return resources


def fetch_metadata_csv_resources(base_url: str) -> list[CsvResource]:
    """Fetch URLs of metadata CSV resources ("assets")."""
    resources = []
    ignored_metadata_files = set(["ogd-smn_meta_datainventory.csv"])

    # Fetch parent resource to get its ID
    # (we could just use the last part of the URL)
    res = fetch_single(base_url)
    parent_id = res["id"]

    for r in fetch_paginated(f"{base_url}/assets"):
        for asset in r.get("assets", []):
            filename = os.path.basename(urlparse(asset["href"]).path)
            if filename in ignored_metadata_files:
                # Ignore irrelevant metadata files.
                continue

            updated = (
                datetime.datetime.fromisoformat(asset["updated"])
                if asset["updated"]
                else None
            )
            # Treat Metadata as historical, i.e. with low update frequency.
            resources.append(
                CsvResource(
                    parent_id=parent_id,
                    href=asset["href"],
                    updated_time=updated,
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
        ("ch.meteoschweiz.ogd-smn", "h"): ds.TABLE_HOURLY_MEASUREMENTS,
        ("ch.meteoschweiz.ogd-smn", "d"): ds.TABLE_DAILY_MEASUREMENTS,
        ("ch.meteoschweiz.ogd-smn", "m"): ds.TABLE_MONTHLY_MEASUREMENTS,
        ("ch.meteoschweiz.ogd-nime", "d"): ds.TABLE_DAILY_MAN_MEASUREMENTS,
        ("ch.meteoschweiz.ogd-nime", "m"): ds.TABLE_MONTHLY_MAN_MEASUREMENTS,
    }
    metadata_tables = {
        "ogd-smn_meta_parameters.csv": ds.sa_table_smn_meta_parameters,
        "ogd-smn_meta_stations.csv": ds.sa_table_smn_meta_stations,
    }

    table_spec = measurement_tables.get((resource.parent_id, resource.interval))
    if table_spec is not None:
        db.insert_csv_data(
            weather_dir,
            engine,
            table_spec=table_spec,
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
            c.updated_time
            and c.status.resource_updated_time
            and c.updated_time > c.status.resource_updated_time
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
            etag, is_modified = download_csv(
                c.href,
                weather_dir,
                etag=c.status.etag if c.status and not force_update else None,
            )
        except Exception as e:
            logger.error("Failed to download %s: %s", c.href, str(e))
            import_queue.put(_NOOP_SENTINEL)  # Inform consumer that we're done.
            return

        if not is_modified:
            # We checked, but the old data is still valid. Update the DB
            # So we don't try again immediately on the next run.
            c.status.mark_not_modified(
                table_updated_time=now, resource_updated_time=c.updated_time
            )
        elif c.status and c.status.id is not None:
            # Update fields in existing UpdateStatus
            c.status.update(
                table_updated_time=now, resource_updated_time=c.updated_time, etag=etag
            )
        else:
            # No existing status or it's a brand-new one: create fresh
            c.status = db.UpdateStatus(
                id=None,  # Leave empty to process as an INSERT.
                href=c.href,
                table_updated_time=now,
                resource_updated_time=c.updated_time,
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
        "--bootstrap-postgres",
        action="store_true",
        default=False,
        help=(
            "If specified, bootstrap the Postgres database (CREATE ROLE etc.) and exit."
            " OGD_POSTGRES_MASTER_SECRET must be set to a JSON containing connection credentials."
        ),
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
        parser.error("--base-dir is required if $OGD_BASE_DIR is not set.")

    if args.postgres_url:
        pgconn = PgConnectionInfo.from_url(args.postgres_url)
    elif "OGD_POSTGRES_ROLE_SECRET" in os.environ:
        # If PG specific env vars are set, use them
        pgconn = PgConnectionInfo.from_env(secret_var="OGD_POSTGRES_ROLE_SECRET")
    elif any(os.getenv(v) for v in ["OGD_POSTGRES_URL", "OGD_DB_HOST"]):
        pgconn = PgConnectionInfo.from_env()
    else:
        pgconn = None  # => local sqlite

    if args.bootstrap_postgres:
        master_secret = os.getenv("OGD_POSTGRES_MASTER_SECRET")
        if None in (pgconn, master_secret):
            parser.error(
                "$OGD_POSTGRES_MASTER_SECRET and pgconn must be set when running --bootstrap-postgres"
            )
        bootstrap_postgres(pgconn, master_secret=master_secret)
        return

    started_time = time.time()

    if pgconn:
        postgres_url = pgconn.url()
        logger.info("Connecting to postgres DB at %s", postgres_url)
        engine = sa.create_engine(postgres_url, echo=args.verbose)
    else:
        db_path = os.path.join(base_dir, dc.DATABASE_FILENAME)
        logger.info("Connecting to sqlite DB at %s", db_path)
        engine = sa.create_engine(f"sqlite:///{db_path}", echo=args.verbose)

    logger.info("Using DB engine '%s'", engine.name)

    if args.recreate_views:
        # Only recreate views, then exit.
        logger.info("Recreating views only...")
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
        ds.metadata.drop_all(bind=engine)

    if force_update:
        logger.info("--force-update was set, forcing data update for all assets")

    # Determine insert mode.
    insert_mode = "insert_missing"
    if force_recreate:
        insert_mode = "append"
    elif force_update:
        insert_mode = "merge"

    # Create tables if needed.
    ds.metadata.create_all(engine)

    # Fetch CSV URLs.
    csvs = []
    csvs.extend(fetch_data_csv_resources(OGD_SNM_URL, args.csv_filter))
    csvs.extend(fetch_data_csv_resources(OGD_NIME_URL, args.csv_filter))
    csvs.extend(fetch_metadata_csv_resources(OGD_SNM_URL))
    csvs.extend(fetch_metadata_csv_resources(OGD_NIME_URL))

    statuses = db.read_update_status(engine)
    logger.debug("Fetched %d update statuses from DB", len(statuses))

    # Assign statuses to corresponding CsvResource.
    status_map = {s.href: s for s in statuses}
    for c in csvs:
        c.status = status_map.get(c.href)

    # Now download CSVs concurrently and feed them to the importer thread
    # via a queue:
    import_queue = queue.SimpleQueue()
    # Count how many actual DB imports were done. If zero, don't recreate views.
    import_count = 0
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
            if resource.status.is_not_modified():
                # Not Modified => Just update timestamps.
                with engine.begin() as conn:
                    db.save_update_status(conn, resource.status)
            else:
                import_into_db(engine, base_dir, resource, insert_mode=insert_mode)
                import_count += 1

        # Consume futures (they're all done)
        for fut in as_completed(futures):
            fut.result()  # Raise exception if any occurred (indicates programming error)

    # Recreate materialized views
    if import_count > 0:
        logger.info("Recreating materialized views...")
        db.recreate_views(engine)
    else:
        logger.info("No new CSV files imported, won't update views.")

    logger.info("Done. DB %s updated in %.1fs.", engine.url, time.time() - started_time)


if __name__ == "__main__":
    main()
