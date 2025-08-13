import datetime
from pydantic import BaseModel, ConfigDict
from datetime import date


class LocalizedString(BaseModel):
    de: str
    fr: str
    it: str
    en: str

    @classmethod
    def from_nullable(
        cls,
        de: str | None = None,
        fr: str | None = None,
        it: str | None = None,
        en: str | None = None,
    ) -> "LocalizedString":
        return cls(de=de or "", fr=fr or "", it=it or "", en=en or "")

    model_config = ConfigDict(frozen=True)


class Station(BaseModel):
    abbr: str
    name: str
    canton: str
    typ: str | None = None
    exposition: LocalizedString | None = None
    url: LocalizedString | None = None
    height_masl: float | None = None
    coordinates_wgs84_lat: float | None = None
    coordinates_wgs84_lon: float | None = None
    temperature_min_date: date | None = None
    temperature_max_date: date | None = None
    precipitation_min_date: date | None = None
    precipitation_max_date: date | None = None


class NearbyStation(BaseModel):
    abbr: str
    name: str
    canton: str
    distance_km: float
    height_diff: float


class StationStats(BaseModel):
    """Weather data statistics for the requested period."""

    first_date: date
    last_date: date
    period: str
    annual_temp_increase: float | None = None
    coldest_year: int | None = None
    coldest_year_temp: float | None = None
    warmest_year: int | None = None
    warmest_year_temp: float | None = None
    driest_year: int | None = None
    driest_year_precip_mm: float | None = None
    wettest_year: int | None = None
    wettest_year_precip_mm: float | None = None


class StationSummary(BaseModel):
    station: Station
    stats: StationStats


class VariableStats(BaseModel):
    min_value: float
    min_value_date: date
    mean_value: float
    max_value: float
    max_value_date: date
    p10_value: float
    p25_value: float
    median_value: float
    p75_value: float
    p90_value: float
    value_sum: float
    value_count: int
    source_granularity: str


class StationPeriodStats(BaseModel):
    """Commonly used summary stats for a given period."""

    start_date: date
    end_date: date
    variable_stats: dict[str, VariableStats]


class StationInfo(BaseModel):
    """Station information and summary statistics."""

    station: Station
    # Summary statistics for the 1991-2020 period.
    ref_1991_2020_stats: StationPeriodStats

    nearby_stations: list[NearbyStation]


class MeasurementsRow(BaseModel):
    reference_timestamp: datetime.datetime
    measurements: list[float]


class ColumnInfo(BaseModel):
    name: str
    display_name: str = ""
    description: str = ""
    dtype: str = ""


class StationMeasurementsData(BaseModel):
    """Generic holder of measurement data for a single station."""

    station_abbr: str
    rows: list[MeasurementsRow]
    columns: list[ColumnInfo]


class StationYearHighlights(BaseModel):
    first_frost_day: datetime.date | None = None
    last_frost_day: datetime.date | None = None
    max_daily_temp_range: float | None = None
    max_daily_temp_range_date: datetime.date | None = None
    max_daily_sunshine_hours: float | None = None
    max_daily_sunshine_hours_date: datetime.date | None = None
    snow_days: float | None = None
    max_snow_depth_cm: float | None = None


class ServerOptions(BaseModel):
    base_dir: str
    sanitized_postgres_url: str = ""
    start_time: datetime.datetime


class DBTableStats(BaseModel):
    schema_name: str  # Should be just "schema", but BaseModel already uses that field.
    table: str
    total_size: str  # usage in human-readable format
    total_bytes: int  # usage in bytes


class ServerStatus(BaseModel):
    current_time_utc: datetime.datetime
    db_engine: str
    options: ServerOptions
    db_table_stats: list[DBTableStats] | None = None  # only for postgres
