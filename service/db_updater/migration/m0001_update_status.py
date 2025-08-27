import logging
import os
import sqlalchemy as sa
from urllib.parse import urlparse

from service.charts.db import schema as ds

from .migration import DBMigration

logger = logging.getLogger("migration")

_MEASUREMENT_TABLES = {
    ("ch.meteoschweiz.ogd-smn", "h"): "ogd_smn_hourly",
    ("ch.meteoschweiz.ogd-smn", "d"): "ogd_smn_daily",
    ("ch.meteoschweiz.ogd-smn", "m"): "ogd_smn_monthly",
    ("ch.meteoschweiz.ogd-smn", "y"): "ogd_smn_annual",
    ("ch.meteoschweiz.ogd-nime", "d"): "ogd_nime_daily",
    ("ch.meteoschweiz.ogd-nime", "m"): "ogd_nime_monthly",
    ("ch.meteoschweiz.ogd-nime", "y"): "ogd_nime_annual",
    ("ch.meteoschweiz.ogd-nbcn", "d"): "ogd_nbcn_daily",
    ("ch.meteoschweiz.ogd-nbcn", "m"): "ogd_nbcn_monthly",
    ("ch.meteoschweiz.ogd-nbcn", "y"): "ogd_nbcn_annual",
}

_METADATA_TABLES = {
    "ogd-smn_meta_parameters.csv": "ogd_smn_meta_parameters",
    "ogd-smn_meta_stations.csv": "ogd_smn_meta_stations",
    "ogd-nime_meta_parameters.csv": "ogd_nime_meta_parameters",
    "ogd-nime_meta_stations.csv": "ogd_nime_meta_stations",
    "ogd-nbcn_meta_parameters.csv": "ogd_smn_meta_parameters",
    "ogd-nbcn_meta_stations.csv": "ogd_nbcn_meta_stations",
}


class UpdateStatusAddDestinationTable(DBMigration, migration_id="m0001_update_status"):
    """Adds the 'destination_table' column to the update_status table and populates it."""

    def _destination_table(self, href: str):
        # Examples:
        # "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/fre/ogd-smn_fre_h_historical_2000-2009.csv"
        # "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nime/mes/ogd-nime_mes_y.csv"
        # "https://data.geo.admin.ch/ch.meteoschweiz.ogd-nbcn/ogd-nbcn_meta_parameters.csv"
        u = urlparse(href)
        fname = os.path.basename(u.path)

        if fname in _METADATA_TABLES:
            return _METADATA_TABLES[fname]

        for (parent_id, hdmy), table_name in _MEASUREMENT_TABLES.items():
            if parent_id in href and (f"{hdmy}.csv" in fname or f"_{hdmy}_" in fname):
                return table_name

        raise ValueError(f"Cannot determine destination table for href {href}")

    def execute(self, engine: sa.Engine):
        if engine.name != "postgresql":
            raise ValueError(
                f"m0001_update_status: can only run on postgresql, got {engine.name}"
            )

        table_name = ds.sa_table_update_status.name
        dest_table_col = "destination_table"

        inspector = sa.inspect(engine)
        columns = inspector.get_columns(table_name)
        if dest_table_col in columns:
            logger.info(f"m0001_update_status: column {dest_table_col} already exists.")
            return

        with engine.begin() as conn:
            # Create column
            add_column_ddl = f"""
                ALTER TABLE {table_name}
                ADD COLUMN {dest_table_col} text
            """
            conn.execute(sa.text(add_column_ddl))

            # Populate column
            result = (
                conn.execute(
                    sa.text(f"SELECT id, href FROM {ds.sa_table_update_status.name}")
                )
                .mappings()
                .all()
            )
            for row in result:
                update_sql = f"""
                    UPDATE {table_name}
                    SET {dest_table_col} = :dest_table
                    WHERE id = :id
                """
                conn.execute(
                    sa.text(update_sql),
                    {
                        "id": row["id"],
                        "dest_table": self._destination_table(row["href"]),
                    },
                )
