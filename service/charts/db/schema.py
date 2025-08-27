"""Contains database schema definitions."""

import pandas as pd
import sqlalchemy as sa

from service.charts.base.errors import SchemaValidationError, SchemaColumnMismatchInfo

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
        dc.TEMP_HOM_DAILY_MIN,
        dc.TEMP_HOM_DAILY_MEAN,
        dc.TEMP_HOM_DAILY_MAX,
        dc.TEMP_HOM_DAILY_MIN_DEV_FROM_NORM_9120,
        dc.TEMP_HOM_DAILY_MEAN_DEV_FROM_NORM_9120,
        dc.TEMP_HOM_DAILY_MAX_DEV_FROM_NORM_9120,
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
        dc.SNOW_DEPTH_MAN_MONTHLY_CM,
        dc.FRESH_SNOW_MAN_MONTHLY_CM,
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
        dc.TEMP_HOM_MONTHLY_MIN,
        dc.TEMP_HOM_MONTHLY_MEAN,
        dc.TEMP_HOM_MONTHLY_MAX,
        dc.TEMP_HOM_MONTHLY_MEAN_OF_DAILY_MAX,
        dc.TEMP_HOM_MONTHLY_MEAN_OF_DAILY_MIN,
        dc.FROST_DAYS_HOM_MONTHLY_COUNT,
        dc.ICE_DAYS_HOM_MONTHLY_COUNT,
        dc.TROPICAL_NIGHTS_HOM_MONTHLY_COUNT,
        # dc.SUMMER_DAYS_HOM_MONTHLY_COUNT,
        dc.HEAT_DAYS_HOM_MONTHLY_COUNT,
        dc.VERY_HOT_DAYS_HOM_MONTHLY_COUNT,
        dc.PRECIP_HOM_MONTHLY_MM,
        dc.WIND_SPEED_HOM_MONTHLY_MEAN,
        dc.ATM_PRESSURE_HOM_MONTHLY_MEAN,
        dc.SUNSHINE_HOM_MONTHLY_MINUTES,
        dc.GLOBAL_RADIATION_HOM_MONTHLY_MEAN,
        dc.HEATING_DEGREE_DAYS_SIA_HOM_MONTHLY_SUM,
        dc.COOLING_DEGREE_DAYS_HOM_MONTHLY_SUM,
    ],
)

# This table is not used at the moment, we only include it
# for consistency, so all measurement types have d/m/y tables.
TABLE_ANNUAL_MEASUREMENTS = DataTableSpec(
    name="ogd_smn_annual",
    primary_key=[
        "station_abbr",
        "reference_timestamp",
    ],
    date_format={
        "reference_timestamp": "%d.%m.%Y %H:%M",
    },
    measurements=[
        dc.PRECIP_ANNUAL_MM,
    ],
)

TABLE_ANNUAL_MAN_MEASUREMENTS = DataTableSpec(
    name="ogd_nime_annual",
    primary_key=[
        "station_abbr",
        "reference_timestamp",
    ],
    date_format={
        "reference_timestamp": "%d.%m.%Y %H:%M",
    },
    measurements=[
        dc.PRECIP_ANNUAL_MM,
        dc.FRESH_SNOW_MAN_ANNUAL_CM,
        dc.SNOW_DEPTH_MAN_ANNUAL_CM,
    ],
)

TABLE_ANNUAL_HOM_MEASUREMENTS = DataTableSpec(
    name="ogd_nbcn_annual",
    primary_key=[
        "station_abbr",
        "reference_timestamp",
    ],
    date_format={
        "reference_timestamp": "%d.%m.%Y %H:%M",
    },
    measurements=[
        dc.TEMP_HOM_ANNUAL_MIN,
        dc.TEMP_HOM_ANNUAL_MEAN,
        dc.TEMP_HOM_ANNUAL_MAX,
        dc.TEMP_HOM_ANNUAL_MEAN_OF_DAILY_MIN,
        dc.TEMP_HOM_ANNUAL_MEAN_OF_DAILY_MAX,
        dc.FROST_DAYS_HOM_ANNUAL_COUNT,
        dc.ICE_DAYS_HOM_ANNUAL_COUNT,
        dc.TROPICAL_NIGHTS_HOM_ANNUAL_COUNT,
        dc.SUMMER_DAYS_HOM_ANNUAL_COUNT,
        dc.HEAT_DAYS_HOM_ANNUAL_COUNT,
        dc.VERY_HOT_DAYS_HOM_ANNUAL_COUNT,
        dc.PRECIP_HOM_ANNUAL_MM,
        dc.WIND_SPEED_HOM_ANNUAL_MEAN,
        dc.ATM_PRESSURE_HOM_ANNUAL_MEAN,
        dc.SUNSHINE_HOM_ANNUAL_MINUTES,
        dc.GLOBAL_RADIATION_HOM_ANNUAL_MEAN,
        dc.HEATING_DEGREE_DAYS_SIA_HOM_ANNUAL_SUM,
        dc.COOLING_DEGREE_DAYS_HOM_ANNUAL_SUM,
    ],
)


def sa_table_meta_stations(station_type: str) -> sa.Table:
    return sa.Table(
        f"ogd_{station_type}_meta_stations",
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


sa_table_smn_meta_stations = sa_table_meta_stations("smn")
sa_table_nime_meta_stations = sa_table_meta_stations("nime")
sa_table_nbcn_meta_stations = sa_table_meta_stations("nbcn")


def sa_table_meta_parameters(station_type: str) -> sa.Table:
    return sa.Table(
        f"ogd_{station_type}_meta_parameters",
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


sa_table_smn_meta_parameters = sa_table_meta_parameters("smn")
sa_table_nime_meta_parameters = sa_table_meta_parameters("nime")
sa_table_nbcn_meta_parameters = sa_table_meta_parameters("nbcn")

sa_table_update_status = sa.Table(
    "update_status",
    metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("href", sa.Text, unique=True, nullable=False),
    sa.Column("table_updated_time", sa.Text, nullable=False),
    sa.Column("resource_updated_time", sa.Text),
    sa.Column("etag", sa.Text),
    sa.Column("destination_table", sa.Text),
)

sa_table_update_log = sa.Table(
    "update_log",
    metadata,
    sa.Column("update_time", sa.Text, primary_key=True),
    sa.Column("imported_files_count", sa.Integer),
    sa.Column("args", sa.Text),
)

# Derived tables / materialized views.
# To distinguish them from SoT data and mark them as derived,
# we prefix them all by "x_"


sa_table_x_all_meta_parameters = sa.Table(
    f"x_all_meta_parameters",
    metadata,
    sa.Column("dataset", sa.Text, primary_key=True),
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

sa_table_x_station_var_availability = sa.Table(
    "x_station_var_availability",
    metadata,
    sa.Column("station_abbr", sa.Text, primary_key=True),
    sa.Column("variable", sa.Text, primary_key=True),
    sa.Column("dataset", sa.Text, primary_key=True),
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
    # Data summaries from the smn daily table:
    sa.Column("tre200d0_count", sa.Integer, nullable=False),
    sa.Column("tre200d0_min_date", sa.Text),
    sa.Column("tre200d0_max_date", sa.Text),
    sa.Column("rre150d0_count", sa.Integer, nullable=False),
    sa.Column("rre150d0_min_date", sa.Text),
    sa.Column("rre150d0_max_date", sa.Text),
    # Data summaries from the nbcn (homogenous data) monthly table:
    sa.Column("ths200m0_count", sa.Integer, nullable=False),
    sa.Column("ths200m0_min_date", sa.Text),
    sa.Column("ths200m0_max_date", sa.Text),
    sa.Column("rhs150m0_count", sa.Integer, nullable=False),
    sa.Column("rhs150m0_min_date", sa.Text),
    sa.Column("rhs150m0_max_date", sa.Text),
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


def validate_schema(
    engine: sa.Engine, allow_missing_tables=True, ignore_derived_tables=True
):
    """
    Validates that the database schema matches the SQLAlchemy MetaData.

    Args:
        engine: the sqlalchemy engine to use
        allow_missing: if True, do not fail if a table is missing in the database,
            but present in the schema (e.g., if it will be created during an update).
        ignore_derived_tables: if True, derived ("x_*") tables are not checked.
            Derived tables are typically regenerated, so there is no need to validate them.

    Raises:
        SchemaValidationError: If there is a mismatch.
    """
    inspector = sa.inspect(engine)
    db_tables = inspector.get_table_names()

    # Step 1: Check for missing tables in the database
    missing_tables = set()
    for table_name in metadata.tables:
        if table_name not in db_tables:
            missing_tables.add(table_name)

    if not allow_missing_tables:
        raise SchemaValidationError(
            f"Tables {missing_tables} are defined in the schema "
            "but do not exist in the database.",
            missing_tables=missing_tables,
        )

    # Step 2: Check for missing/mismatched columns in existing tables
    mismatched_columns = []
    for table_name, table in metadata.tables.items():
        if table_name in missing_tables:
            continue
        if ignore_derived_tables and table_name.startswith("x_"):
            continue

        db_columns_info = inspector.get_columns(table_name)
        db_columns = {col["name"]: col for col in db_columns_info}

        for col_name, column in table.columns.items():
            if col_name not in db_columns:
                mismatched_columns.append(
                    SchemaColumnMismatchInfo(
                        table=table_name, column=col_name, is_missing=True
                    )
                )
                continue

            # Compare column types (this is the trickiest part)
            db_col_type = db_columns[col_name]["type"]
            code_col_type = column.type

            # Using str() comparison is often good enough for common types.
            # For 100% accuracy, you might need more complex dialect-specific logic.
            if not isinstance(db_col_type, type(code_col_type)):
                # A simple fallback for when types don't match directly
                if str(code_col_type).upper() not in str(db_col_type).upper():
                    mismatched_columns.append(
                        SchemaColumnMismatchInfo(
                            table=table_name,
                            column=col_name,
                            info=f"Schema: {code_col_type}, DB: {db_col_type}",
                        )
                    )
                    continue

            # Compare nullability
            if column.nullable != db_columns[col_name]["nullable"]:
                mismatched_columns.append(
                    SchemaColumnMismatchInfo(
                        table=table_name,
                        column=col_name,
                        info=f"Schema: nullable={column.nullable}, DB: nullable={db_columns[col_name]['nullable']}",
                    )
                )
                continue

    if len(mismatched_columns) > 0:
        raise SchemaValidationError(
            f"Column mismatches detected", mismatched_columns=mismatched_columns
        )
