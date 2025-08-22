"""Contains database schema definitions."""

import pandas as pd
import sqlalchemy as sa

from . import constants as dc

# SQLAlchemy pattern: have a global 'metadata' variable that holds all table defs.
metadata = sa.MetaData()


class DataTableSpec:
    """Schema specification for a measurement table + SQLAlchmey Table binding."""

    def __init__(
        self,
        name: str,
        primary_key: list[str],
        measurements: list[str],
        date_format: dict[str, str] | None = None,
        description: str = "",
    ) -> None:
        self.name: str = name
        self.primary_key: list[str] = primary_key
        self.measurements: list[str] = measurements
        self.date_format: dict[str, str] | None = date_format
        self.description: str = description
        # Reference to the sa.Table instance:
        # Ensure this table gets registered in metadata immediately.
        self.sa_table: sa.Table = self._define_sa_table()

    def _define_sa_table(self) -> sa.Table:
        """Converts a DataTableSpec into a SQLAlchemy Core Table definition.

        Registers this table in the global `metadata`.
        """

        # Primary key columns are always TEXT
        pk_columns = [
            sa.Column(col_name, sa.Text, primary_key=True)
            for col_name in self.primary_key
        ]

        # Measurement columns are always REAL: Use 32-bit accuracy is more than enough.
        measurement_columns = [
            sa.Column(col_name, sa.REAL) for col_name in self.measurements
        ]

        return sa.Table(self.name, metadata, *(pk_columns + measurement_columns))

    def columns(self):
        return self.primary_key + self.measurements


# Table definitions

TABLE_HOURLY_MEASUREMENTS = DataTableSpec(
    name="ogd_smn_hourly",
    primary_key=[
        "station_abbr",
        "reference_timestamp",
    ],
    date_format={
        "reference_timestamp": "%d.%m.%Y %H:%M",
    },
    measurements=[
        dc.TEMP_HOURLY_MEAN,
        dc.TEMP_HOURLY_MIN,
        dc.TEMP_HOURLY_MAX,
        dc.PRECIP_HOURLY_MM,
        dc.GUST_PEAK_HOURLY_MAX,
        dc.ATM_PRESSURE_HOURLY_MEAN,
        dc.REL_HUMIDITY_HOURLY_MEAN,
        dc.SUNSHINE_HOURLY_MINUTES,
        dc.WIND_SPEED_HOURLY_MEAN,
        dc.WIND_DIRECTION_HOURLY_MEAN,
    ],
)

TABLE_DAILY_MEASUREMENTS = DataTableSpec(
    name="ogd_smn_daily",
    primary_key=[
        "station_abbr",
        "reference_timestamp",
    ],
    date_format={
        "reference_timestamp": "%d.%m.%Y %H:%M",
    },
    measurements=[
        dc.TEMP_DAILY_MEAN,
        dc.TEMP_DAILY_MIN,
        dc.TEMP_DAILY_MAX,
        dc.PRECIP_DAILY_MM,
        dc.GUST_PEAK_DAILY_MAX,
        dc.ATM_PRESSURE_DAILY_MEAN,
        dc.REL_HUMIDITY_DAILY_MEAN,
        dc.SUNSHINE_DAILY_MINUTES,
        dc.SUNSHINE_DAILY_PCT_OF_MAX,
        dc.WIND_SPEED_DAILY_MEAN,
        dc.WIND_DIRECTION_DAILY_MEAN,
        dc.SNOW_DEPTH_DAILY_CM,
    ],
)

TABLE_DAILY_MAN_MEASUREMENTS = DataTableSpec(
    name="ogd_nime_daily",
    primary_key=[
        "station_abbr",
        "reference_timestamp",
    ],
    date_format={
        "reference_timestamp": "%d.%m.%Y %H:%M",
    },
    measurements=[
        dc.PRECIP_DAILY_MM,
        dc.SNOW_DEPTH_MAN_DAILY_CM,
        dc.FRESH_SNOW_MAN_DAILY_CM,
    ],
)

TABLE_DAILY_HOM_MEASUREMENTS = DataTableSpec(
    name="ogd_nbcn_daily",
    primary_key=[
        "station_abbr",
        "reference_timestamp",
    ],
    date_format={
        "reference_timestamp": "%d.%m.%Y %H:%M",
    },
    measurements=[
        dc.TEMP_DAILY_HOM_MIN,
        dc.TEMP_DAILY_HOM_MEAN,
        dc.TEMP_DAILY_HOM_MAX,
    ],
)

TABLE_MONTHLY_MEASUREMENTS = DataTableSpec(
    name="ogd_smn_monthly",
    primary_key=[
        "station_abbr",
        "reference_timestamp",
    ],
    date_format={
        "reference_timestamp": "%d.%m.%Y %H:%M",
    },
    measurements=[
        dc.TEMP_MONTHLY_MIN,
        dc.TEMP_MONTHLY_MEAN,
        dc.TEMP_MONTHLY_MAX,
        dc.PRECIP_MONTHLY_MM,
        dc.WIND_SPEED_MONTHLY_MEAN,
        dc.GUST_PEAK_MONTHLY_MAX,
        dc.ATM_PRESSURE_MONTHLY_MEAN,
        dc.REL_HUMIDITY_MONTHLY_MEAN,
        dc.SUNSHINE_MONTHLY_MINUTES,
    ],
)

TABLE_MONTHLY_MAN_MEASUREMENTS = DataTableSpec(
    name="ogd_nime_monthly",
    primary_key=[
        "station_abbr",
        "reference_timestamp",
    ],
    date_format={
        "reference_timestamp": "%d.%m.%Y %H:%M",
    },
    measurements=[
        dc.PRECIP_MONTHLY_MM,
    ],
)

TABLE_MONTHLY_HOM_MEASUREMENTS = DataTableSpec(
    name="ogd_nbcn_monthly",
    primary_key=[
        "station_abbr",
        "reference_timestamp",
    ],
    date_format={
        "reference_timestamp": "%d.%m.%Y %H:%M",
    },
    measurements=[
        dc.TEMP_DAILY_HOM_MIN,
        dc.TEMP_DAILY_HOM_MEAN,
        dc.TEMP_DAILY_HOM_MAX,
    ],
)


sa_table_smn_meta_stations = sa.Table(
    "ogd_smn_meta_stations",
    metadata,
    sa.Column("station_abbr", sa.Text, primary_key=True),
    sa.Column("station_name", sa.Text),
    sa.Column("station_canton", sa.Text),
    sa.Column("station_wigos_id", sa.Text),
    sa.Column("station_type_de", sa.Text),
    sa.Column("station_type_fr", sa.Text),
    sa.Column("station_type_it", sa.Text),
    sa.Column("station_type_en", sa.Text),
    sa.Column("station_dataowner", sa.Text),
    sa.Column("station_data_since", sa.Text),
    sa.Column("station_height_masl", sa.REAL),
    sa.Column("station_height_barometer_masl", sa.REAL),
    sa.Column("station_coordinates_lv95_east", sa.REAL),
    sa.Column("station_coordinates_lv95_north", sa.REAL),
    sa.Column("station_coordinates_wgs84_lat", sa.REAL),
    sa.Column("station_coordinates_wgs84_lon", sa.REAL),
    sa.Column("station_exposition_de", sa.Text),
    sa.Column("station_exposition_fr", sa.Text),
    sa.Column("station_exposition_it", sa.Text),
    sa.Column("station_exposition_en", sa.Text),
    sa.Column("station_url_de", sa.Text),
    sa.Column("station_url_fr", sa.Text),
    sa.Column("station_url_it", sa.Text),
    sa.Column("station_url_en", sa.Text),
)

sa_table_smn_meta_parameters = sa.Table(
    "ogd_smn_meta_parameters",
    metadata,
    sa.Column("parameter_shortname", sa.Text, primary_key=True),
    sa.Column("parameter_description_de", sa.Text),
    sa.Column("parameter_description_fr", sa.Text),
    sa.Column("parameter_description_it", sa.Text),
    sa.Column("parameter_description_en", sa.Text),
    sa.Column("parameter_group_de", sa.Text),
    sa.Column("parameter_group_fr", sa.Text),
    sa.Column("parameter_group_it", sa.Text),
    sa.Column("parameter_group_en", sa.Text),
    sa.Column("parameter_granularity", sa.Text),
    sa.Column("parameter_decimals", sa.Integer),
    sa.Column("parameter_datatype", sa.Text),
    sa.Column("parameter_unit", sa.Text),
)


sa_table_update_status = sa.Table(
    "update_status",
    metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("href", sa.Text, unique=True, nullable=False),
    sa.Column("table_updated_time", sa.Text, nullable=False),
    sa.Column("resource_updated_time", sa.Text),
    sa.Column("etag", sa.Text),
)


# Derived tables / materialized views.
# To distinguish them from SoT data and mark them as derived,
# we prefix them all by "x_"


sa_table_x_station_var_availability = sa.Table(
    "x_station_var_availability",
    metadata,
    sa.Column("station_abbr", sa.Text, primary_key=True),
    sa.Column("variable", sa.Text, primary_key=True),
    sa.Column("value_count", sa.Integer, nullable=False),
    sa.Column("min_date", sa.Text),
    sa.Column("max_date", sa.Text),
)

sa_table_x_station_data_summary = sa.Table(
    "x_station_data_summary",
    metadata,
    sa.Column("station_abbr", sa.Text, primary_key=True),
    sa.Column("station_name", sa.Text),
    sa.Column("station_canton", sa.Text),
    sa.Column("station_wigos_id", sa.Text),
    sa.Column("station_type_en", sa.Text),
    sa.Column("station_exposition_de", sa.Text),
    sa.Column("station_exposition_fr", sa.Text),
    sa.Column("station_exposition_it", sa.Text),
    sa.Column("station_exposition_en", sa.Text),
    sa.Column("station_url_de", sa.Text),
    sa.Column("station_url_fr", sa.Text),
    sa.Column("station_url_it", sa.Text),
    sa.Column("station_url_en", sa.Text),
    sa.Column("station_dataowner", sa.Text),
    sa.Column("station_data_since", sa.Text),
    sa.Column("station_height_masl", sa.REAL),
    sa.Column("station_coordinates_wgs84_lat", sa.REAL),
    sa.Column("station_coordinates_wgs84_lon", sa.REAL),
    sa.Column("tre200d0_count", sa.Integer, nullable=False),
    sa.Column("tre200d0_min_date", sa.Text),
    sa.Column("tre200d0_max_date", sa.Text),
    sa.Column("rre150d0_count", sa.Integer, nullable=False),
    sa.Column("rre150d0_min_date", sa.Text),
    sa.Column("rre150d0_max_date", sa.Text),
)

sa_table_x_nearby_stations = sa.Table(
    "x_nearby_stations",
    metadata,
    sa.Column("from_station_abbr", sa.Text, primary_key=True),
    sa.Column("from_station_name", sa.Text),
    sa.Column("from_station_canton", sa.Text),
    sa.Column("to_station_abbr", sa.Text, primary_key=True),
    sa.Column("to_station_name", sa.Text),
    sa.Column("to_station_canton", sa.Text),
    sa.Column("distance_km", sa.REAL),
    sa.Column("height_diff", sa.REAL),
)


class VarSummaryStatsTable:

    def __init__(self, name: str, time_slice: str) -> None:
        self.name = name
        if time_slice == "all":
            self.time_slicer = VarSummaryStatsTable._ts_all
        elif time_slice == "month":
            self.time_slicer = VarSummaryStatsTable._ts_month
        else:
            raise ValueError(f"invalid time slice: {time_slice}")
        self.sa_table: sa.Table = self._define_station_var_summary_stats_table(name)

    def _define_station_var_summary_stats_table(self, table_name: str) -> sa.Table:
        return sa.Table(
            table_name,
            metadata,
            sa.Column("agg_name", sa.Text, primary_key=True),
            sa.Column("station_abbr", sa.Text, primary_key=True),
            sa.Column("variable", sa.Text, primary_key=True),
            sa.Column("time_slice", sa.Text, primary_key=True),
            sa.Column("source_granularity", sa.Text),
            # Min/max including date of occurrence
            sa.Column("min_value", sa.REAL),
            sa.Column("min_value_date", sa.Text),
            sa.Column("mean_value", sa.REAL),
            sa.Column("max_value", sa.REAL),
            sa.Column("max_value_date", sa.Text),
            # Percentiles (10, 25, 50, 75, 90)
            sa.Column("p10_value", sa.REAL),
            sa.Column("p25_value", sa.REAL),
            sa.Column("median_value", sa.REAL),
            sa.Column("p75_value", sa.REAL),
            sa.Column("p90_value", sa.REAL),
            # Sum
            sa.Column("value_sum", sa.REAL),
            # Count
            sa.Column("value_count", sa.Integer),
        )

    @classmethod
    def _ts_month(cls, d: pd.DataFrame) -> pd.Series:
        return pd.to_datetime(d["reference_timestamp"]).dt.month.map(dc.ts_month)

    @classmethod
    def _ts_all(cls, d: pd.DataFrame) -> pd.Series:
        return pd.Series(["*"] * len(d.index), index=d.index)


var_summary_stats_all = VarSummaryStatsTable(
    name="x_station_var_summary_stats",
    time_slice="all",
)


var_summary_stats_month = VarSummaryStatsTable(
    name="x_station_var_summary_stats_month",
    time_slice="month",
)
