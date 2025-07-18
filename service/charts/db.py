import sqlite3
from . import models

from typing import Optional
import sqlite3
from . import models  # Station BaseModel


def read_stations(
    conn: sqlite3.Connection,
    cantons: list[str] | None = None,
    exclude_empty: bool = True,
) -> list[models.Station]:
    """Returns all stations matching the given criteria.

    - cantons: optional list of canton codes to filter
    - exclude_empty: if True, skips stations with no temp/precip data
    """

    sql = """
        SELECT station_abbr, station_name, station_canton
        FROM ogd_smn_station_data_summary
    """
    filters: list[str] = []
    params: list = []

    # Canton filter
    if cantons:
        placeholders = ",".join("?" for _ in cantons)
        filters.append(f"station_canton IN ({placeholders})")
        params.extend(cantons)

    # Exclude stations with no data
    if exclude_empty:
        filters.append("(tre200d0_count > 0 AND rre150d0_count > 0)")

    # Combine filters
    if filters:
        sql += " WHERE " + " AND ".join(filters)

    sql += " ORDER BY station_abbr"

    cur = conn.execute(sql, params)
    rows = cur.fetchall()

    return [
        models.Station(
            abbr=row["station_abbr"],
            name=row["station_name"],
            canton=row["station_canton"],
        )
        for row in rows
    ]
