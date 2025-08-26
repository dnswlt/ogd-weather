from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import logging
import os
import queue
import re
import sys
import time
from pydantic import BaseModel
import requests
import sqlalchemy as sa
from urllib.parse import urlparse

from service.charts.base import logging_config as _  # configure logging

from service.charts import db
from service.charts.db import schema as ds


logger = logging.getLogger("ogd")

_NOOP_SENTINEL = object()

OGD_SNM_URL = (
    "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn"
)

OGD_NIME_URL = (
    "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-nime"
)

OGD_NBCN_URL = (
    "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-nbcn"
)


UPDATE_STATUS_TABLE_NAME = "update_status"

MEASUREMENT_TABLES = {
    ("ch.meteoschweiz.ogd-smn", "h"): ds.TABLE_HOURLY_MEASUREMENTS,
    ("ch.meteoschweiz.ogd-smn", "d"): ds.TABLE_DAILY_MEASUREMENTS,
    ("ch.meteoschweiz.ogd-smn", "m"): ds.TABLE_MONTHLY_MEASUREMENTS,
    ("ch.meteoschweiz.ogd-smn", "y"): ds.TABLE_ANNUAL_MEASUREMENTS,
    ("ch.meteoschweiz.ogd-nime", "d"): ds.TABLE_DAILY_MAN_MEASUREMENTS,
    ("ch.meteoschweiz.ogd-nime", "m"): ds.TABLE_MONTHLY_MAN_MEASUREMENTS,
    ("ch.meteoschweiz.ogd-nime", "y"): ds.TABLE_ANNUAL_MAN_MEASUREMENTS,
    ("ch.meteoschweiz.ogd-nbcn", "d"): ds.TABLE_DAILY_HOM_MEASUREMENTS,
    ("ch.meteoschweiz.ogd-nbcn", "m"): ds.TABLE_MONTHLY_HOM_MEASUREMENTS,
    ("ch.meteoschweiz.ogd-nbcn", "y"): ds.TABLE_ANNUAL_HOM_MEASUREMENTS,
}

METADATA_TABLES = {
    "ogd-smn_meta_parameters.csv": ds.sa_table_smn_meta_parameters,
    "ogd-smn_meta_stations.csv": ds.sa_table_smn_meta_stations,
    "ogd-nime_meta_parameters.csv": ds.sa_table_nime_meta_parameters,
    "ogd-nime_meta_stations.csv": ds.sa_table_nime_meta_stations,
    "ogd-nbcn_meta_parameters.csv": ds.sa_table_nbcn_meta_parameters,
    "ogd-nbcn_meta_stations.csv": ds.sa_table_nbcn_meta_stations,
}


class HrefMatch(BaseModel):
    """Utility class to represent a href match result."""

    interval: str  # one of ("h", "d", "m")
    frequency: str  # one of ("historical", "recent", "now")
    years: tuple[int, int] | None = None


def match_csv_resource(href: str, filter_re: str | None = None) -> HrefMatch | None:
    if filter_re is not None and not re.search(filter_re, href):
        return None
    # TODO: Re-enable _now_ data at some point.
    # It had a few missing points recently,
    # we don't reconcile with recent data yet, and we're currently
    # focusing on historical data anyway.

    mo = re.search(
        r".*_(?P<interval>d|h|m|y)(_(?P<freq>historical|historical_(?P<years>\d+-\d+)|recent|now__DISABLED__))?.csv$",
        href,
    )
    if mo is None:
        return None

    interval = mo.group("interval")
    freq = mo.group("freq")
    if not freq:
        # Interpret missing suffix as historical data (happens e.g. for "m")
        freq = "historical"
    elif freq.startswith("historical"):
        freq = "historical"  # Trim suffix years.
    years_str = mo.group("years")
    years = tuple(map(int, years_str.split("-"))) if years_str else None

    return HrefMatch(
        interval=interval,
        frequency=freq,
        years=years,
    )


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
        if match is None:
            continue
        if (parent_id, match.interval) not in MEASUREMENT_TABLES:
            logger.info("Ignoring %s: no matching measurement table exists", href)
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
    ignored_metadata_files = set(
        [
            "ogd-smn_meta_datainventory.csv",
            "ogd-nime_meta_datainventory.csv",
            "ogd-nbcn_meta_datainventory.csv",
        ]
    )

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

    table_spec = MEASUREMENT_TABLES.get((resource.parent_id, resource.interval))
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
        table = METADATA_TABLES.get(csv_file)
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


def run_import(
    engine: sa.Engine,
    base_dir: str,
    insert_mode: str,
    force_update: bool,
    csv_filter: str | None,
):
    started_time = time.time()

    # Create tables if needed.
    ds.metadata.create_all(engine)

    # Fetch CSV URLs.
    csvs: list[CsvResource] = []
    csvs.extend(fetch_data_csv_resources(OGD_SNM_URL, csv_filter))
    csvs.extend(fetch_data_csv_resources(OGD_NIME_URL, csv_filter))
    csvs.extend(fetch_data_csv_resources(OGD_NBCN_URL, csv_filter))
    csvs.extend(fetch_metadata_csv_resources(OGD_SNM_URL))
    csvs.extend(fetch_metadata_csv_resources(OGD_NIME_URL))
    csvs.extend(fetch_metadata_csv_resources(OGD_NBCN_URL))

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

    db.add_update_log_entry(
        engine,
        db.UpdateLogEntry(
            update_time=datetime.datetime.fromtimestamp(
                started_time, tz=datetime.timezone.utc
            ),
            imported_files_count=import_count,
            args=sys.argv,
        ),
    )

    logger.info(
        "Done. DB %s updated in %.1fs.", engine.name, time.time() - started_time
    )


def run_recreate_views(engine: sa.Engine):
    started_time = time.time()
    # Only recreate views, then exit.
    logger.info("Recreating views only...")
    db.recreate_views(engine)
    logger.info("Recreated materialized views in %.1fs.", time.time() - started_time)
