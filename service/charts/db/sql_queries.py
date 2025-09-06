"""Contains SQL queries for various use cases."""

import datetime
import sqlalchemy as sa

from service.charts.base.dates import utc_timestr, utc_datestr

from . import constants as dc
from . import schema as ds


def ts_local(dialect: sa.Dialect, col: str) -> str:
    """Returns the local date and time in Europe/Zurich for a timestamp.

    The returned timestamp does NOT contain a time zone offset suffix.

    Example: '2024-07-31 15:30:59' for '2024-07-31 13:30:59Z'
    """
    if dialect.name == "postgresql":
        return f"to_char({col}::timestamptz AT TIME ZONE 'Europe/Zurich', 'YYYY-MM-DD HH24:MI:SS')"
    elif dialect.name == "sqlite":
        return f"datetime({col}, 'localtime')"
    else:
        raise ValueError(f"Unsupported dialect: {dialect.name}")


def ts_month(col: str) -> str:
    """Returns the month of a timestamp.

    Example: '07' for '2024-07-31 13:30:59Z'
    """
    return f"SUBSTR({col}, 6, 2)"


def ts_yearmonth(col: str) -> str:
    """Returns the year and month of a timestamp.

    Example: '2024-07' for '2024-07-31 13:30:59Z'
    """
    return f"SUBSTR({col}, 1, 7)"


def ts_date(col: str) -> str:
    """Returns the date part of a timestamp.

    Example: '2024-07-31' for '2024-07-31 13:30:59Z'
    """
    return f"SUBSTR({col}, 1, 10)"


def ts_hourminute(col: str) -> str:
    """Returns the hour and minute of a timestamp.

    Example: '13:30' for '2024-07-31 13:30:59Z'
    """
    return f"SUBSTR({col}, 12, 5)"


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


def insert_into_x_ogd_smn_daily_derived(
    dialect: sa.Dialect,
    from_date: datetime.datetime,
    to_date: datetime.datetime,
) -> sa.TextClause:
    """Returns the INSERT statement with bound params to populate daily derived data.

    NOTE that the SQL statement has to use the *end* times of the hourly intervals
    (e.g. '08:00' represents the interval 07:00-08:00), since that's the meaning
    of the hourly reference_timestamp values.
    """
    sql = f"""
        INSERT INTO {ds.sa_table_x_ogd_smn_daily_derived.name}
        WITH 
            LocalHourlyMeasurements AS (
                SELECT
                    station_abbr,
                    {ts_local(dialect, "reference_timestamp")} AS local_timestamp,
                    {dc.PRECIP_HOURLY_MM},
                    {dc.VAPOR_PRESSURE_HOURLY_MEAN},
                    {dc.WIND_SPEED_HOURLY_MEAN},
                    {dc.GUST_PEAK_HOURLY_MAX}
                FROM {ds.TABLE_HOURLY_MEASUREMENTS.name}
                WHERE reference_timestamp >= :from_date
                    AND reference_timestamp < :to_date
            ),
            DaytimeValues AS (
                SELECT
                    *
                FROM LocalHourlyMeasurements
                WHERE 
                    (
                        CASE
                            WHEN {ts_month("local_timestamp")} IN ('05', '06', '07', '08')
                                THEN {ts_hourminute("local_timestamp")} BETWEEN '08:00' AND '22:00'
                            WHEN {ts_month("local_timestamp")} IN ('03', '04', '09', '10')
                                THEN {ts_hourminute("local_timestamp")} BETWEEN '08:00' AND '21:00'
                            ELSE {ts_hourminute("local_timestamp")} BETWEEN '08:00' AND '18:00'
                        END
                    )
            )
        SELECT
            station_abbr,
            {ts_date("local_timestamp")} AS reference_timestamp,
            SUM({dc.PRECIP_HOURLY_MM}) AS {dc.DX_PRECIP_DAYTIME_DAILY_MM},
            MAX({dc.VAPOR_PRESSURE_HOURLY_MEAN}) AS {dc.DX_VAPOR_PRESSURE_DAYTIME_DAILY_MAX_OF_HOURLY_MEAN},
            MAX({dc.WIND_SPEED_HOURLY_MEAN}) AS {dc.DX_WIND_SPEED_DAYTIME_DAILY_MAX_OF_HOURLY_MEAN},
            MAX({dc.GUST_PEAK_HOURLY_MAX}) AS {dc.DX_GUST_PEAK_DAYTIME_DAILY_MAX}
        FROM DaytimeValues
        GROUP BY station_abbr, reference_timestamp
        """

    return sa.text(sql).bindparams(
        from_date=utc_timestr(from_date),
        to_date=utc_timestr(to_date),
    )


def insert_into_x_wind_stats_monthly(
    dialect: sa.Dialect,
    from_date: datetime.datetime,
    to_date: datetime.datetime,
) -> sa.TextClause:
    """Returns the INSERT statement with bound params to populate monthly wind stats."""

    sql = f"""
        INSERT INTO {ds.sa_table_x_wind_stats_monthly.name}
        WITH
            LocalHourlyMeasurements AS (
                SELECT
                    station_abbr,
                    {ts_local(dialect, "reference_timestamp")} AS local_timestamp,
                    {dc.WIND_SPEED_HOURLY_MEAN},
                    {dc.GUST_PEAK_HOURLY_MAX},
                    {dc.WIND_DIRECTION_HOURLY_MEAN}
                FROM {ds.TABLE_HOURLY_MEASUREMENTS.name}
                WHERE
                    reference_timestamp >= :from_date
                    AND reference_timestamp < :to_date
            ),
            DailyMaxes AS (
                SELECT
                    station_abbr,
                    {ts_date("local_timestamp")} AS local_date,
                    MAX({dc.WIND_SPEED_HOURLY_MEAN} * 3.6) AS max_hourly_avg_wind_kmh,
                    MAX({dc.GUST_PEAK_HOURLY_MAX} * 3.6) AS max_gust_kmh,
                    COUNT(*) AS value_count
                FROM LocalHourlyMeasurements
                GROUP BY station_abbr, local_date
            ),
            StationDayCountsMonthly AS (
                SELECT
                    station_abbr,
                    {ts_yearmonth("local_date")} AS local_month,
                    SUM(CASE WHEN max_hourly_avg_wind_kmh >= 20 THEN 1 ELSE 0 END) AS moderate_breeze_days,
                    SUM(CASE WHEN max_gust_kmh >= 39 THEN 1 ELSE 0 END) AS strong_breeze_days,
                    SUM(value_count) AS value_count
                FROM DailyMaxes
                GROUP BY station_abbr, local_month
            ),
            StationGustFactorMonthly AS (
                SELECT
                    station_abbr,
                    {ts_yearmonth("local_timestamp")} AS local_month,
                    AVG({dc.GUST_PEAK_HOURLY_MAX} / {dc.WIND_SPEED_HOURLY_MEAN}) AS gust_factor
                FROM LocalHourlyMeasurements
                WHERE {dc.WIND_SPEED_HOURLY_MEAN} > 0
                GROUP BY station_abbr, local_month
            ),
            StationWindDirectionsLong AS (
                SELECT
                    station_abbr,
                    {ts_yearmonth("local_timestamp")} AS local_month,
                    -- 45 degree buckets, where -22.5 to 22.5 is N.
                    (CAST(FLOOR(({dc.WIND_DIRECTION_HOURLY_MEAN} + 22.5) / 45.0) AS INTEGER) % 8) AS bucket_index,
                    COUNT(*) AS value_count
                FROM LocalHourlyMeasurements
                WHERE {dc.WIND_DIRECTION_HOURLY_MEAN} IS NOT NULL
                GROUP BY station_abbr, local_month, bucket_index
            ),
            StationWindDirectionsMonthly AS (
                SELECT
                    station_abbr,
                    local_month,
                    SUM(CASE WHEN bucket_index = 0 THEN value_count ELSE 0 END) AS {dc.DX_WIND_DIR_N_COUNT},
                    SUM(CASE WHEN bucket_index = 1 THEN value_count ELSE 0 END) AS {dc.DX_WIND_DIR_NE_COUNT},
                    SUM(CASE WHEN bucket_index = 2 THEN value_count ELSE 0 END) AS {dc.DX_WIND_DIR_E_COUNT},
                    SUM(CASE WHEN bucket_index = 3 THEN value_count ELSE 0 END) AS {dc.DX_WIND_DIR_SE_COUNT},
                    SUM(CASE WHEN bucket_index = 4 THEN value_count ELSE 0 END) AS {dc.DX_WIND_DIR_S_COUNT},
                    SUM(CASE WHEN bucket_index = 5 THEN value_count ELSE 0 END) AS {dc.DX_WIND_DIR_SW_COUNT},
                    SUM(CASE WHEN bucket_index = 6 THEN value_count ELSE 0 END) AS {dc.DX_WIND_DIR_W_COUNT},
                    SUM(CASE WHEN bucket_index = 7 THEN value_count ELSE 0 END) AS {dc.DX_WIND_DIR_NW_COUNT},
                    SUM(value_count) AS {dc.DX_WIND_DIR_TOTAL_COUNT}
                FROM StationWindDirectionsLong
                GROUP BY station_abbr, local_month
            )
        SELECT
            station_abbr,
            (local_month || '-01') AS reference_timestamp,
            SDC.moderate_breeze_days AS {dc.DX_MODERATE_BREEZE_DAYS_MONTHLY_COUNT},
            SDC.strong_breeze_days AS {dc.DX_STRONG_BREEZE_DAYS_MONTHLY_COUNT},
            SGF.gust_factor AS {dc.DX_GUST_FACTOR_MONTHLY_MEAN},
            SDC.value_count AS {dc.DX_VALUE_COUNT},
            SWD.{dc.DX_WIND_DIR_N_COUNT},
            SWD.{dc.DX_WIND_DIR_NE_COUNT},
            SWD.{dc.DX_WIND_DIR_E_COUNT},
            SWD.{dc.DX_WIND_DIR_SE_COUNT},
            SWD.{dc.DX_WIND_DIR_S_COUNT},
            SWD.{dc.DX_WIND_DIR_SW_COUNT},
            SWD.{dc.DX_WIND_DIR_W_COUNT},
            SWD.{dc.DX_WIND_DIR_NW_COUNT},
            SWD.{dc.DX_WIND_DIR_TOTAL_COUNT}
        FROM StationDayCountsMonthly AS SDC
        JOIN StationGustFactorMonthly SGF USING (station_abbr, local_month)
        LEFT JOIN StationWindDirectionsMonthly AS SWD USING (station_abbr, local_month)
    """

    return sa.text(sql).bindparams(
        from_date=utc_timestr(from_date),
        to_date=utc_timestr(to_date),
    )


def daily_nice_day_metrics(
    station_abbr: str,
    from_date: datetime.date,
    to_date: datetime.date,
) -> sa.TextClause:
    sql = f"""
        SELECT
            station_abbr,
            reference_timestamp,
            D.{dc.TEMP_DAILY_MAX},
            D.{dc.SUNSHINE_DAILY_PCT_OF_MAX},
            DX.{dc.DX_PRECIP_DAYTIME_DAILY_MM},
            DX.{dc.DX_VAPOR_PRESSURE_DAYTIME_DAILY_MAX_OF_HOURLY_MEAN},
            DX.{dc.DX_WIND_SPEED_DAYTIME_DAILY_MAX_OF_HOURLY_MEAN},
            DX.{dc.DX_GUST_PEAK_DAYTIME_DAILY_MAX}
        FROM {ds.TABLE_DAILY_MEASUREMENTS.name} AS D
        JOIN {ds.sa_table_x_ogd_smn_daily_derived.name} AS DX
            USING (station_abbr, reference_timestamp)
        WHERE
            station_abbr = :station_abbr
            AND reference_timestamp >= :from_date
            AND reference_timestamp < :to_date
        ORDER BY reference_timestamp
    """
    return sa.text(sql).bindparams(
        station_abbr=station_abbr,
        from_date=utc_datestr(from_date),
        to_date=utc_datestr(to_date),
    )
