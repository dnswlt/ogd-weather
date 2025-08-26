import argparse
import logging
import os
import sqlalchemy as sa

from service.charts.base import logging_config as _  # configure logging

from service.charts.base.errors import SchemaValidationError
from service.charts.db.dbconn import PgConnectionInfo
from service.charts.db import schema as ds
from service.charts.db import constants as dc

from . import bootstrap
from . import ogd

logger = logging.getLogger("db_updater")


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
        bootstrap.bootstrap_postgres(pgconn, master_secret=master_secret)
        return

    if pgconn:
        postgres_url = pgconn.url()
        logger.info("Connecting to postgres DB at %s", pgconn.sanitized_url())
        engine = sa.create_engine(postgres_url, echo=args.verbose)
    else:
        db_path = os.path.join(base_dir, dc.SQLITE_DB_FILENAME)
        logger.info("Connecting to sqlite DB at %s", db_path)
        engine = sa.create_engine(f"sqlite:///{db_path}", echo=args.verbose)

    logger.info("Using DB engine '%s'", engine.name)

    try:
        ds.validate_schema(engine=engine, allow_missing_tables=True)
        logger.info("Successfully validated the DB schema.")
    except SchemaValidationError as e:
        logger.error(
            "Schema mismatch detected: %s."
            " Consider running with --force-recreate. "
            " Note that this will DROP ALL DATA from the database.",
            str(e),
        )
        return

    if args.recreate_views:
        ogd.run_recreate_views(engine)
        return

    force_update: bool = args.force_update
    force_recreate: bool = args.force_recreate

    # Drop old data if requested.
    if force_recreate:
        logger.info("Dropping existing data from all tables.")
        ds.metadata.drop_all(bind=engine)

    if force_update:
        logger.info("--force-update was set, forcing data update for all assets")

    # Determine insert mode.
    insert_mode = "insert_missing"
    if force_recreate:
        insert_mode = "append"
    elif force_update:
        insert_mode = "merge"

    ogd.run_import(
        engine=engine,
        base_dir=base_dir,
        insert_mode=insert_mode,
        force_update=force_update,
        csv_filter=args.csv_filter,
    )


if __name__ == "__main__":
    main()
