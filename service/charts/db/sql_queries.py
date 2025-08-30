"""Contains SQL queries for various use cases."""

import datetime
import sqlalchemy as sa

from service.charts.base.dates import utc_timestr

from . import constants as dc
from . import schema as ds


def psql_total_bytes(user: str) -> sa.TextClause:
    sql = sa.text(
        f"""
        SELECT
            n.nspname AS schema,
            c.relname AS table,
            pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
            pg_total_relation_size(c.oid) AS total_bytes
        FROM
            pg_class c
        JOIN
            pg_namespace n ON n.oid = c.relnamespace
        WHERE
            c.relkind = 'r'  -- regular table
            AND pg_catalog.pg_get_userbyid(c.relowner) = :user
        ORDER BY
            pg_total_relation_size(c.oid) DESC;
        """
    )
    return sql.bindparams(user=user)


def insert_into_monthly_wind_stats(
    from_date: datetime.datetime,
    to_date: datetime.datetime,
    date_range: str,
) -> sa.TextClause:
    """Returns the INSERT INTO statement and params to populate monthly wind stats."""
    sql = sa.text(
        f"""
        INSERT INTO {ds.sa_table_x_monthly_wind_stats.name}
        WITH
            DailyMaxes AS (
                SELECT
                    station_abbr,
                    SUBSTR(reference_timestamp, 1, 10) AS day,
                    MAX({dc.WIND_SPEED_HOURLY_MEAN} * 3.6) AS max_hourly_avg_wind_kmh,
                    MAX({dc.GUST_PEAK_HOURLY_MAX} * 3.6) AS max_gust_kmh,
                    COUNT(*) AS value_count
                FROM
                    {ds.TABLE_HOURLY_MEASUREMENTS.name}
                WHERE
                    reference_timestamp >= :from_date
                    AND reference_timestamp < :to_date
                GROUP BY station_abbr, day
            ),
            StationDayCountsAnnual AS (
                SELECT
                    station_abbr,
                    CAST(SUBSTR(day, 1, 4) AS INTEGER) AS year,
                    CAST(SUBSTR(day, 6, 2) AS INTEGER) AS month,
                    SUM(CASE WHEN max_hourly_avg_wind_kmh >= 20 THEN 1 ELSE 0 END) AS moderate_breeze_days,
                    SUM(CASE WHEN max_gust_kmh >= 39 THEN 1 ELSE 0 END) AS strong_breeze_days,
                    SUM(value_count) AS value_count
                FROM DailyMaxes
                GROUP BY station_abbr, year, month
            ),
            StationDayCounts AS (
                SELECT
                    station_abbr,
                    month,
                    AVG(moderate_breeze_days) AS moderate_breeze_days,
                    AVG(strong_breeze_days) AS strong_breeze_days,
                    SUM(value_count) AS value_count
                FROM StationDayCountsAnnual
                GROUP BY station_abbr, month
            ),
            StationGustFactor AS (
                SELECT
                    station_abbr,
                    CAST(SUBSTR(reference_timestamp, 6, 2) AS INTEGER) AS month,
                    AVG({dc.GUST_PEAK_HOURLY_MAX} / {dc.WIND_SPEED_HOURLY_MEAN}) AS gust_factor
                FROM {ds.TABLE_HOURLY_MEASUREMENTS.name}
                WHERE
                    reference_timestamp >= :from_date
                    AND reference_timestamp < :to_date
                    AND {dc.WIND_SPEED_HOURLY_MEAN} > 0
                GROUP BY station_abbr, month
            ),
            StationWindDirectionsLong AS (
                SELECT
                    station_abbr,
                    CAST(SUBSTR(reference_timestamp, 6, 2) AS INTEGER) AS month,
                    -- 45 degree buckets, where -22.5 to 22.5 is N.
                    (CAST(FLOOR(({dc.WIND_DIRECTION_HOURLY_MEAN} + 22.5) / 45.0) AS INTEGER) % 8) AS bucket_index,
                    COUNT(*) AS value_count
                FROM {ds.TABLE_HOURLY_MEASUREMENTS.name}
                WHERE
                    reference_timestamp >= :from_date
                    AND reference_timestamp < :to_date
                    AND {dc.WIND_DIRECTION_HOURLY_MEAN} IS NOT NULL
                GROUP BY station_abbr, month, bucket_index
            ),
            StationWindDirections AS (
                SELECT
                    station_abbr,
                    month,
                    SUM(CASE WHEN bucket_index = 0 THEN value_count ELSE 0 END) AS wind_dir_n_count,
                    SUM(CASE WHEN bucket_index = 1 THEN value_count ELSE 0 END) AS wind_dir_ne_count,
                    SUM(CASE WHEN bucket_index = 2 THEN value_count ELSE 0 END) AS wind_dir_e_count,
                    SUM(CASE WHEN bucket_index = 3 THEN value_count ELSE 0 END) AS wind_dir_se_count,
                    SUM(CASE WHEN bucket_index = 4 THEN value_count ELSE 0 END) AS wind_dir_s_count,
                    SUM(CASE WHEN bucket_index = 5 THEN value_count ELSE 0 END) AS wind_dir_sw_count,
                    SUM(CASE WHEN bucket_index = 6 THEN value_count ELSE 0 END) AS wind_dir_w_count,
                    SUM(CASE WHEN bucket_index = 7 THEN value_count ELSE 0 END) AS wind_dir_nw_count,
                    SUM(value_count) AS wind_dir_total_count
                FROM StationWindDirectionsLong
                GROUP BY station_abbr, month
            )
        SELECT
            :date_range AS date_range,
            station_abbr,
            month,
            SDC.moderate_breeze_days,
            SDC.strong_breeze_days,
            SGF.gust_factor,
            SDC.value_count,
            SWD.wind_dir_n_count,
            SWD.wind_dir_ne_count,
            SWD.wind_dir_e_count,
            SWD.wind_dir_se_count,
            SWD.wind_dir_s_count,
            SWD.wind_dir_sw_count,
            SWD.wind_dir_w_count,
            SWD.wind_dir_nw_count,
            SWD.wind_dir_total_count
        FROM StationDayCounts AS SDC
        JOIN StationGustFactor SGF USING (station_abbr, month)
        LEFT JOIN StationWindDirections AS SWD USING (station_abbr, month)
    """
    )
    return sql.bindparams(
        date_range=date_range,
        from_date=utc_timestr(from_date),
        to_date=utc_timestr(to_date),
    )
